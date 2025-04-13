"""
Database connection utilities.
"""

import psycopg2
from .config import DB_CONFIG

def get_db_connection():
    """Creates and returns a PostgreSQL database connection and cursor.
    
    Returns:
        tuple: (connection, cursor) tuple with active database connections
    """
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        port=DB_CONFIG["port"]
    )
    cursor = conn.cursor()
    return conn, cursor
