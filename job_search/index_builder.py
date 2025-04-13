"""
FAISS index builder for job embeddings.
"""

import numpy as np
import faiss
import io
import traceback
import tempfile
import os
from .db import get_db_connection
from .embedding import get_long_text_embedding
from .config import FAISS_INDEX_NAME

def serialize_faiss_index(index):
    """Serialize a FAISS index to bytes.
    
    Args:
        index: FAISS index to serialize
        
    Returns:
        bytes: Serialized index
    """
    try:
        # Create a temporary file to write the index
        fd, path = tempfile.mkstemp()
        try:
            # Close the file descriptor
            os.close(fd)
            
            # Write the index to the temporary file
            faiss.write_index(index, path)
            
            # Read the file content
            with open(path, 'rb') as f:
                serialized_data = f.read()
                
            return serialized_data
        finally:
            # Clean up the temporary file
            os.unlink(path)
    except Exception as e:
        print(f"Error serializing FAISS index: {e}")
        traceback.print_exc()
        raise

def create_job_embeddings():
    """Calculate and store embeddings for all jobs in the database.
    
    Returns:
        int: Number of jobs processed
    """
    conn, cursor = get_db_connection()

    try:
        # Check if the embedding column exists
        cursor.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name='job_postings' AND column_name='embedding';
        """)
        
        if not cursor.fetchone():
            # Add embedding column if it doesn't exist
            cursor.execute("ALTER TABLE job_postings ADD COLUMN embedding FLOAT[];")
            conn.commit()
        
        # Fetch job descriptions that need embeddings
        cursor.execute("""
            SELECT job_id, description FROM job_postings 
            WHERE description IS NOT NULL AND embedding IS NULL;
        """)
        jobs = cursor.fetchall()
        
        # Process and store embeddings
        for i, (job_id, description) in enumerate(jobs):
            print(f"Processing job {i+1}/{len(jobs)}: ID {job_id}")
            
            try:
                # Compute embedding
                embedding = get_long_text_embedding(description)
                embedding_list = np.array(embedding).tolist()

                # Store embedding in the database
                cursor.execute(
                    "UPDATE job_postings SET embedding = %s WHERE job_id = %s;",
                    (embedding_list, job_id)
                )
                
                # Commit every 100 jobs to avoid large transactions
                if (i + 1) % 100 == 0:
                    conn.commit()
                    print(f"Committed batch of 100 job embeddings")
            except Exception as e:
                print(f"Error processing job {job_id}: {e}")
                continue
        
        # Final commit
        conn.commit()
        return len(jobs)
    except Exception as e:
        conn.rollback()
        print(f"Error creating job embeddings: {e}")
        traceback.print_exc()
        return 0
    finally:
        cursor.close()
        conn.close()

def build_faiss_index():
    """Build and store FAISS index from job embeddings.
    
    Returns:
        bool: True if successful, False otherwise
    """
    conn, cursor = get_db_connection()
    
    try:
        # Create necessary tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_indices (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE,
                index_data BYTEA,
                dimension INTEGER,
                num_vectors INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faiss_job_mapping (
                faiss_index_name VARCHAR(255),
                vector_position INTEGER,
                job_id INTEGER,
                PRIMARY KEY (faiss_index_name, vector_position)
            );
        """)
        
        # Fetch job embeddings
        cursor.execute("SELECT job_id, embedding FROM job_postings WHERE embedding IS NOT NULL;")
        job_data = cursor.fetchall()
        
        if not job_data:
            print("No embeddings found in job_postings table.")
            return False
        
        print(f"Found {len(job_data)} job embeddings in database.")
        
        # Extract job IDs and embeddings
        job_ids = []
        embeddings = []
        
        for job_id, embedding_data in job_data:
            try:
                # Check if embedding is a string and convert it properly
                if isinstance(embedding_data, str):
                    # Remove brackets and split by comma
                    embedding_str = embedding_data.strip('[]')
                    embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                    vector = np.array(embedding_values, dtype='float32')
                else:
                    # If it's already an array or list, just use it
                    vector = np.array(embedding_data, dtype='float32')
                
                job_ids.append(job_id)
                embeddings.append(vector)
            except Exception as e:
                print(f"Error processing embedding for job {job_id}: {e}")
                continue
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        print(f"Embeddings array shape: {embeddings_array.shape}")
        
        # Get embedding dimension
        d = embeddings_array.shape[1]
        num_vectors = len(embeddings_array)
        
        # Create FAISS index
        # For very small datasets (fewer than 39 vectors), use a simple index
        if num_vectors < 39:
            print(f"Small dataset ({num_vectors} vectors), using IndexFlatL2")
            index = faiss.IndexFlatL2(d)
        else:
            # For larger datasets, use IVFPQ
            # Create quantizer
            quantizer = faiss.IndexFlatL2(d)
            
            # Calculate appropriate number of clusters (Recommended: num_vectors / 39)
            # At least 4 clusters but no more than 256
            num_clusters = min(256, max(4, int(num_vectors / 39)))
            
            print(f"Creating IVFPQ index with {num_clusters} clusters")
            
            # Ensure m (number of subquantizers) is a divisor of d
            m = 8  # Default number of subquantizers
            while d % m != 0 and m > 1:
                m -= 1  # Reduce until it's a divisor
                
            print(f"Using {m} subquantizers for dimension {d}")
            
            # Create the IVFPQ index
            nbits = 8  # Bits per subquantizer (usually 8)
            index = faiss.IndexIVFPQ(quantizer, d, num_clusters, m, nbits)
            
            # Train the index
            print("Training IVFPQ index...")
            index.train(embeddings_array)
            
            # Set nprobe for better recall/speed tradeoff
            # Higher values = better recall but slower search
            index.nprobe = min(num_clusters, 8)
            print(f"Set nprobe={index.nprobe}")
        
        # Add vectors to the index
        print("Adding vectors to index...")
        index.add(embeddings_array)
        
        # Serialize the index
        print("Serializing index...")
        serialized_index = serialize_faiss_index(index)
        
        # Clear previous mappings
        cursor.execute(f"DELETE FROM faiss_job_mapping WHERE faiss_index_name = '{FAISS_INDEX_NAME}';")
        
        # Insert new mappings
        print("Storing job ID mappings...")
        for pos, job_id in enumerate(job_ids):
            cursor.execute(
                "INSERT INTO faiss_job_mapping (faiss_index_name, vector_position, job_id) VALUES (%s, %s, %s);",
                (FAISS_INDEX_NAME, pos, job_id)
            )
        
        # Store index in database
        print("Storing FAISS index in database...")
        cursor.execute("""
            INSERT INTO faiss_indices (name, index_data, dimension, num_vectors)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) 
            DO UPDATE SET 
                index_data = EXCLUDED.index_data,
                dimension = EXCLUDED.dimension,
                num_vectors = EXCLUDED.num_vectors,
                created_at = CURRENT_TIMESTAMP;
        """, (FAISS_INDEX_NAME, serialized_index, d, len(embeddings)))
        
        conn.commit()
        print(f"Successfully built and saved FAISS index with {len(embeddings)} job embeddings.")
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error building FAISS index: {e}")
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """Main function to run the index builder."""
    # Step 1: Create embeddings for jobs that don't have them
    print("=== Creating Job Embeddings ===")
    num_processed = create_job_embeddings()
    print(f"Processed {num_processed} jobs")
    
    # Step 2: Build FAISS index
    print("\n=== Building FAISS Index ===")
    success = build_faiss_index()
    
    if success:
        print("\nIndex build complete! You can now use the job search functionality.")
    else:
        print("\nIndex build failed. Please check the errors above.")

if __name__ == "__main__":
    main()
