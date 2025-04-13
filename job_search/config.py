"""
Configuration settings for the job search application.
"""

# PostgreSQL connection details
DB_CONFIG = {
    "host": "localhost",
    "database": "job_data",
    "user": "postgres",
    "password": "Eenadu@1",
    "port": "5433"
}

# FAISS index name used throughout the application
FAISS_INDEX_NAME = "job_matching_index"

# Embedding dimension
EMBEDDING_DIM = 768
