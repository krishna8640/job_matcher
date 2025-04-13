import requests
import os
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from datetime import datetime
import json
import schedule
import time
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "job_fetch.log")

# Configure logging
logger = logging.getLogger("job_fetcher")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Load environment variables from .env file
load_dotenv(dotenv_path=".env.new")

# Add these lines after load_dotenv() to clean any problematic values
DB_HOST = os.getenv("DB_HOST", "localhost").strip()
DB_PORT = os.getenv("DB_PORT", "5433").strip().split("#")[0].strip()  # Remove comments
DB_NAME = os.getenv("DB_NAME", "job_data").strip()
DB_USER = os.getenv("DB_USER", "postgres").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()

JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY", "").strip().split("#")[0].strip()



# Jooble API credentials
JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY")

# Define the job categories to focus on
STEM_JOBS = [
    "data scientist", 
    "software engineer",
    "chemical engineer", 
    "mechanical engineer",
    "biomedical engineer",
    "electrical engineer",
    "environmental scientist",
    "mathematician",
    "statistician"
]

RESEARCH_JOBS = [
    "research scientist", 
    "research assistant", 
    "lab technician",
    "postdoctoral researcher",
    "research associate",
    "clinical researcher"
]

HEALTHCARE_JOBS = [
    "physician", 
    "nurse practitioner", 
    "registered nurse",
    "pharmacist",
    "physical therapist",
    "occupational therapist",
    "medical technologist",
    "radiologist",
    "healthcare analyst"
]

# Locations to search in (can be customized)
LOCATIONS = [
    "New York", 
    "California", 
    "Texas", 
    "Massachusetts", 
    "Washington",
    "Remote"
]

def connect_to_db():
    """Connect to PostgreSQL database and return connection object"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        logger.info("Successfully connected to the database!")
        return conn
    except psycopg2.Error as e:
        logger.error(f"Unable to connect to the database: {e}")
        return None

def fetch_jooble_jobs(keywords=None, location=None, page=1, per_page=20):
    """Fetch job listings from Jooble API"""
    # Jooble API endpoint
    api_url = f"https://jooble.org/api/{JOOBLE_API_KEY}"
    
    # Build request body
    request_data = {}
    
    if keywords and keywords.strip():
        request_data["keywords"] = keywords.strip()
    if location and location.strip():
        request_data["location"] = location.strip()
        
    # Add pagination parameters
    request_data["page"] = page
    request_data["pageSize"] = per_page
    
    try:
        logger.info(f"Sending API request: keywords={keywords}, location={location}, page={page}")
        # Make the POST request
        response = requests.post(api_url, json=request_data)
        
        # For debugging
        logger.debug(f"Request URL: {api_url}")
        logger.debug(f"Request Body: {json.dumps(request_data)}")
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse and return JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching jobs: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response content: {e.response.text[:1000]}")  # Print first 1000 chars of error
        return None

def insert_jobs_into_db(conn, jobs_data):
    """Insert job data into the job_postings table, mapping Jooble API fields to database columns"""
    if not jobs_data or "jobs" not in jobs_data or not jobs_data["jobs"]:
        logger.warning("No jobs found or invalid response")
        return 0
    
    cursor = conn.cursor()
    jobs_inserted = 0
    jobs_skipped = 0
    
    try:
        for job in jobs_data['jobs']:
            # Convert Jooble id to string if it's an integer
            job_id = str(job.get('id', ''))
            
            # Parse location into components
            location = job.get('location', '')
            location_parts = location.split(',')
            location_short = location_parts[0].strip() if location_parts else ''
            state_code = location_parts[1].strip() if len(location_parts) > 1 else ''
            
            # Map Jooble API fields to your database columns
            job_data = {
                'job_id': job_id,
                'job_title': job.get('title', ''),
                'url': job.get('link', ''),
                'company_name': job.get('company', ''),
                'description': job.get('snippet', ''),
                'location_short': location_short,
                'location_long': location,
                'state_code': state_code,
                'date_posted': parse_date(job.get('updated', '')),
                'job_category': determine_job_category(job.get('title', ''), job.get('snippet', ''))
            }
            
            # Check if the job already exists
            cursor.execute("SELECT job_id FROM job_postings WHERE job_id = %s", (job_data['job_id'],))
            existing_job = cursor.fetchone()
            
            if existing_job:
                jobs_skipped += 1
                continue
            
            # Build the SQL insert statement
            fields = job_data.keys()
            placeholders = ['%s'] * len(fields)
            
            sql = f"""
            INSERT INTO job_postings ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
            """
            
            cursor.execute(sql, list(job_data.values()))
            jobs_inserted += 1
            
        conn.commit()
        logger.info(f"Inserted {jobs_inserted} new jobs, skipped {jobs_skipped} existing jobs")
        return jobs_inserted
        
    except psycopg2.Error as e:
        logger.error(f"Error inserting jobs: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()

def determine_job_category(title, description):
    """Determine the job category based on title and description"""
    title_lower = title.lower()
    desc_lower = description.lower()
    
    # Check for STEM jobs
    for job in STEM_JOBS:
        if job.lower() in title_lower or job.lower() in desc_lower:
            return "STEM"
    
    # Check for Research jobs
    for job in RESEARCH_JOBS:
        if job.lower() in title_lower or job.lower() in desc_lower:
            return "Research"
    
    # Check for Healthcare jobs
    for job in HEALTHCARE_JOBS:
        if job.lower() in title_lower or job.lower() in desc_lower:
            return "Healthcare"
    
    # If no match found
    return "Other"

def parse_date(date_str):
    """Convert API date string to database format"""
    if not date_str:
        return None
    
    try:
        # Format appears to be: "2025-04-06T00:00:00.0000000"
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            # Try a more generic approach if the above fails
            from dateutil import parser
            return parser.parse(date_str)
        except:
            return None

def fetch_and_store_jobs(keywords=None, location=None, max_pages=5, per_page=20):
    """Fetch jobs from Jooble API and store them in the database"""
    conn = connect_to_db()
    if not conn:
        logger.error("Failed to connect to the database")
        return
    
    try:
        total_jobs_inserted = 0
        
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page} for {keywords} in {location}...")
            jobs_data = fetch_jooble_jobs(keywords, location, page, per_page)
            
            if not jobs_data or "jobs" not in jobs_data or not jobs_data["jobs"]:
                logger.info(f"No more results found after page {page-1}")
                break
            
            # Get total count if this is the first page
            if page == 1 and "totalCount" in jobs_data:
                total_count = jobs_data["totalCount"]
                logger.info(f"Total jobs available: {total_count}")
                
            jobs_inserted = insert_jobs_into_db(conn, jobs_data)
            total_jobs_inserted += jobs_inserted
            
            # Print more detailed information
            logger.info(f"Page {page}: Found {len(jobs_data['jobs'])} jobs, inserted {jobs_inserted}")
            
            # If we received fewer results than requested, we're probably at the end
            if len(jobs_data["jobs"]) < per_page:
                logger.info("Received fewer results than requested, ending search.")
                break
                
        logger.info(f"Total jobs inserted: {total_jobs_inserted}")
    
    finally:
        conn.close()

def run_job_search():
    """Run job search for all predefined categories and locations"""
    logger.info("Starting scheduled job search...")
    
    # Combine all job types into one list
    all_job_types = STEM_JOBS + RESEARCH_JOBS + HEALTHCARE_JOBS
    
    # Iterate through each job type and location
    for job_type in all_job_types:
        for location in LOCATIONS:
            try:
                logger.info(f"Searching for {job_type} jobs in {location}")
                fetch_and_store_jobs(
                    keywords=job_type,
                    location=location,
                    max_pages=3,  # Limit to 3 pages per search to avoid API limits
                    per_page=20
                )
                # Sleep between requests to avoid hitting API rate limits
                time.sleep(5)  
            except Exception as e:
                logger.error(f"Error processing {job_type} in {location}: {e}")
    
    logger.info("Scheduled job search completed successfully")

def setup_database():
    """Ensure database has the necessary table and columns"""
    conn = connect_to_db()
    if not conn:
        logger.error("Failed to connect to the database for setup")
        return False
    
    cursor = conn.cursor()
    
    try:
        # Check if job_category column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='job_postings' AND column_name='job_category';
        """)
        
        if not cursor.fetchone():
            logger.info("Adding job_category column to job_postings table")
            cursor.execute("ALTER TABLE job_postings ADD COLUMN job_category VARCHAR(50);")
            conn.commit()
        
        # Check if the table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'job_postings'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("Creating job_postings table")
            cursor.execute("""
                CREATE TABLE job_postings (
                    job_id VARCHAR(100) PRIMARY KEY,
                    job_title VARCHAR(255),
                    url TEXT,
                    company_name VARCHAR(255),
                    company_domain VARCHAR(255),
                    company_industry VARCHAR(255),
                    company_employee_count VARCHAR(100),
                    location_short VARCHAR(100),
                    location_long VARCHAR(255),
                    state_code VARCHAR(10),
                    latitude FLOAT,
                    longitude FLOAT,
                    description TEXT,
                    date_posted TIMESTAMP,
                    embedding FLOAT[],
                    job_category VARCHAR(50)
                );
            """)
            conn.commit()
            
        return True
    except psycopg2.Error as e:
        logger.error(f"Error setting up database: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def schedule_jobs():
    """Schedule jobs to run at specified intervals"""
    # Ensure database is set up
    if not setup_database():
        logger.error("Failed to set up database. Exiting.")
        return
    
    # Run immediately on startup
    run_job_search()
    
    # Schedule to run daily at specific times
    schedule.every().day.at("06:00").do(run_job_search)
    schedule.every().day.at("14:00").do(run_job_search)
    schedule.every().day.at("22:00").do(run_job_search)
    
    logger.info("Scheduled job search to run at 06:00, 14:00, and 22:00 daily")
    
    # Run the scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    logger.info("Job fetching automation started")
    schedule_jobs()