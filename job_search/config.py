"""
Configuration settings for the job search application.
"""

# PostgreSQL connection details
# PostgreSQL connection details (Render-hosted)
DB_CONFIG = {
    "host": "dpg-cvu00thr0fns73e29j9g-a.oregon-postgres.render.com",
    "database": "job_data_xjal",
    "user": "job_data_xjal_user",
    "password": "f8Qs1P2YfTgHknrV0aCyr1i0UzVUPAaY",
    "port": "5432"
}

# FAISS index name used throughout the application
FAISS_INDEX_NAME = "job_matching_index"

# Embedding dimension
EMBEDDING_DIM = 768
