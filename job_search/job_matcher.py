"""
Job matching functionality using FAISS index.
"""

import faiss
import numpy as np
import tempfile
import os
import traceback
import math
from .db import get_db_connection
from .embedding import get_long_text_embedding
from .config import FAISS_INDEX_NAME

def deserialize_faiss_index(serialized_index):
    """Deserialize a FAISS index from bytes.
    
    Args:
        serialized_index (bytes): Serialized FAISS index
        
    Returns:
        faiss.Index: Deserialized FAISS index
    """
    try:
        # Create a temporary file
        fd, path = tempfile.mkstemp()
        try:
            # Write the serialized data to the temp file
            with os.fdopen(fd, 'wb') as f:
                f.write(serialized_index)
            
            # Read the index from the file
            index = faiss.read_index(path)
            return index
        finally:
            # Clean up
            if os.path.exists(path):
                os.unlink(path)
    except Exception as e:
        print(f"Error deserializing FAISS index: {e}")
        traceback.print_exc()
        raise

def load_faiss_index():
    """Load the FAISS index from the database."""
    conn, cursor = get_db_connection()
    try:
        # Instead of loading the saved index, rebuild a simple one on the fly
        cursor.execute("SELECT job_id, embedding FROM job_postings WHERE embedding IS NOT NULL;")
        job_data = cursor.fetchall()
        
        if not job_data:
            print("No embeddings found in database.")
            return None, {}
        
        # Process embeddings
        job_ids = []
        embeddings = []
        
        for job_id, embedding_data in job_data:
            try:
                # Handle string format
                if isinstance(embedding_data, str):
                    embedding_str = embedding_data.strip('[]')
                    embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                    vector = np.array(embedding_values, dtype='float32')
                else:
                    vector = np.array(embedding_data, dtype='float32')
                
                job_ids.append(job_id)
                embeddings.append(vector)
            except Exception as e:
                print(f"Error processing embedding: {e}")
                continue
                
        if not embeddings:
            return None, {}
            
        # Create ID mapping
        id_mapping = {i: job_id for i, job_id in enumerate(job_ids)}
        
        # Create a simple index
        embeddings_array = np.array(embeddings).astype('float32')
        d = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_array)
        
        return index, id_mapping
    except Exception as e:
        print(f"Error loading/building FAISS index: {e}")
        traceback.print_exc()
        return None, {}
    finally:
        cursor.close()
        conn.close()

def get_job_columns():
    """Get the actual column names from the job_postings table."""
    conn, cursor = get_db_connection()
    try:
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'job_postings';
        """)
        columns = [row[0] for row in cursor.fetchall()]
        print(f"Available columns in job_postings: {columns}")
        return columns
    finally:
        cursor.close()
        conn.close()

def get_job_details(job_ids):
    """Get details for specified job IDs."""
    if not job_ids:
        return []
    
    conn, cursor = get_db_connection()
    try:
        # First check what columns exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'job_postings';
        """)
        available_columns = [row[0] for row in cursor.fetchall()]
        print(f"Available columns: {available_columns}")
        
        # Build the query based on available columns
        select_columns = ["job_id"]  # We know this one exists
        
        # Add other columns if they exist
        column_mapping = {
            "title": "title", 
            "name": "title",  # Alternative name
            "job_title": "title",  # Alternative name
            "company": "company",
            "company_name": "company",  # Alternative name
            "description": "description",
            "job_description": "description",  # Alternative name
            "location_short": "location_short",
            "location_long": "location_long",
            "salary_range": "salary_range",
            "job_type": "job_type",
            "url": "url",
        }
        
        for db_col, result_col in column_mapping.items():
            if db_col.lower() in [col.lower() for col in available_columns]:
                select_columns.append(f"\"{db_col}\" as \"{result_col}\"")
        
        # Create the query
        query = f"""
            SELECT {', '.join(select_columns)}
            FROM job_postings
            WHERE job_id IN ({', '.join(['%s'] * len(job_ids))});
        """
        print(f"Query: {query}")
        
        cursor.execute(query, job_ids)
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in cursor.fetchall():
            job_dict = dict(zip(columns, row))
            # Ensure each result has all expected fields, even if null
            for expected_field in ["title", "company", "description", "location", "salary_range", "job_type"]:
                if expected_field not in job_dict:
                    job_dict[expected_field] = "Not specified"
            results.append(job_dict)
        return results
    except Exception as e:
        print(f"Error fetching job details: {e}")
        traceback.print_exc()
        return []
    finally:
        cursor.close()
        conn.close()

def search_jobs(query_text, top_k=200, page=1, limit=10):
    """
    Search for jobs matching the query text with pagination.
    
    Args:
        query_text: The search query text
        top_k: Maximum number of results to retrieve from FAISS
        page: Page number to return (1-based)
        limit: Number of results per page
        
    Returns:
        dict: Contains results, total count, current page, and total pages
    """
    try:
        # Calculate offset from page number
        offset = (page - 1) * limit

        # Generate embedding for query
        print("Generating embedding for query...")
        query_embedding = get_long_text_embedding(query_text)
        query_np = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Load/build index on the fly
        print("Loading FAISS index...")
        index, id_mapping = load_faiss_index()
        if index is None:
            print("No index available.")
            return {"results": [], "total": 0, "page": page, "total_pages": 0}
        
        # Search
        print(f"Searching for top {top_k} matches...")
        k = min(top_k, len(id_mapping))
        if k == 0:
            return {"results": [], "total": 0, "page": page, "total_pages": 0}
            
        distances, indices = index.search(query_np, k=k)
        
        # Get job IDs
        job_ids = []
        similarity_scores = {}
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx in id_mapping:
                job_id = id_mapping[idx]
                job_ids.append(job_id)
                similarity_scores[job_id] = float(1.0 - min(distances[0][i], 100) / 100)
        
        # Get details
        print(f"Fetching details for {len(job_ids)} jobs...")
        job_details = get_job_details(job_ids)
        
        # Add scores
        for job in job_details:
            job['similarity_score'] = similarity_scores.get(job['job_id'], 0)
        
        # Sort
        job_details.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Apply pagination
        total_results = len(job_details)
        total_pages = max(1, math.ceil(total_results / limit))
        
        # Get the paginated subset
        paginated_results = job_details[offset:offset + limit]
        
        print(f"Found {total_results} matching jobs, returning page {page} with {len(paginated_results)} jobs")
        
        # Return dictionary with all necessary data
        return {
            "results": paginated_results,
            "total": total_results,
            "page": page,
            "total_pages": total_pages
        }
    except Exception as e:
        print(f"Error searching jobs: {e}")
        traceback.print_exc()
        return {"results": [], "total": 0, "page": page, "total_pages": 0}