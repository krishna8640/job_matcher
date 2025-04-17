"""
Job matching functionality using FAISS index.
"""

import faiss
import numpy as np
import math
import traceback
from .db import get_db_connection
from .embedding import get_long_text_embedding
from .config import FAISS_INDEX_NAME
from .index_cache import IndexCache

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
    
    snippet = job_ids[:5] + ["..."] if len(job_ids) > 5 else job_ids
    print(f"Getting details for job IDs: {snippet}")
    
    conn, cursor = get_db_connection()
    try:
        # job_id is stored as TEXT in our table, so bind them as strings
        job_ids_str = [str(j) for j in job_ids]
        if not job_ids_str:
            print("No valid job IDs to query")
            return []
        
        placeholders = ', '.join(['%s'] * len(job_ids_str))
        query = f"""
            SELECT 
                job_id::text,
                job_title,
                company_name,
                description,
                location_short,
                location_long,
                job_category,
                url
            FROM job_postings
            WHERE job_id IN ({placeholders});
        """
        print(f"Executing query for {len(job_ids_str)} job IDs")
        
        # Bind as strings so TEXT column matches
        cursor.execute(query, tuple(job_ids_str))
        
        columns = [desc[0] for desc in cursor.description]
        print(f"Returned columns: {columns}")
        
        results = []
        for row in cursor.fetchall():
            job_dict = dict(zip(columns, row))
            result = {
                "job_id": job_dict["job_id"],
                "title": job_dict.get("job_title", "Not specified"),
                "company": job_dict.get("company_name", "Not specified"),
                "description": job_dict.get("description", "Not specified"),
                "location_short": job_dict.get("location_short", "Not specified"),
                "location_long": job_dict.get("location_long", "Not specified"),
                "job_type": job_dict.get("job_category", "Not specified"),
                "url": job_dict.get("url", "")
            }
            # Combine into a single location field
            result["location"] = (
                result["location_long"] 
                if result["location_long"] != "Not specified" 
                else result["location_short"]
            )
            results.append(result)
        
        print(f"Retrieved {len(results)} jobs from {len(job_ids_str)} requested IDs")
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
    """
    try:
        offset = (page - 1) * limit

        print("Generating embedding for query...")
        query_embedding = get_long_text_embedding(query_text)
        query_np = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        print("Getting FAISS index from cache...")
        index_cache = IndexCache.get_instance()
        if index_cache.index is None:
            print("FAISS index is None! Reloading…")
            index_cache.is_loaded = False
            index_cache.load_index()
        
        print(f"ID mapping contains {len(index_cache.id_mapping)} entries")
        
        k = min(top_k, len(index_cache.id_mapping))
        if k == 0:
            print("Warning: No vectors in ID mapping!")
            return {"results": [], "total": 0, "page": page, "total_pages": 0}
        
        print(f"Searching for top {k} matches…")
        distances, indices = index_cache.search(query_np, k=k)
        print(f"Search returned {indices.shape[1]} results")
        
        job_ids = []
        similarity_scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in index_cache.id_mapping:
                jid = index_cache.id_mapping[idx]
                job_ids.append(jid)
                similarity_scores[jid] = float(1.0 - min(dist, 100) / 100)
        
        print(f"Found {len(job_ids)} valid job IDs to retrieve")
        job_details = get_job_details(job_ids)
        
        for job in job_details:
            job['similarity_score'] = similarity_scores.get(job['job_id'], 0.0)
        
        job_details.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        total_results = len(job_details)
        total_pages = max(1, math.ceil(total_results / limit))
        paginated = job_details[offset:offset + limit]
        
        print(f"Returning page {page}/{total_pages} with {len(paginated)} jobs")
        return {
            "results": paginated,
            "total": total_results,
            "page": page,
            "total_pages": total_pages
        }

    except Exception as e:
        print(f"Error searching jobs: {e}")
        traceback.print_exc()
        return {"results": [], "total": 0, "page": page, "total_pages": 0}

# For direct testing
if __name__ == "__main__":
    sample_ids = ["390379353", "389064627", "388688744"]
    print("Test get_job_details:", get_job_details(sample_ids))
