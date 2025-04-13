import requests
import os
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from datetime import datetime
import time
import schedule
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "adzuna_job_fetch.log")

# Configure logging
logger = logging.getLogger("adzuna_job_fetcher")
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

# Adzuna API credentials
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

# Define job search categories with keywords
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

# Define locations to search in
LOCATIONS = [
    "New York", 
    "California", 
    "Texas", 
    "Massachusetts", 
    "Washington",
    ""  # Empty string for nationwide search
]

# 2. Database Connection Function
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

def fetch_adzuna_jobs(what=None, where=None, category=None, page=1, results_per_page=100, country="us"):
    """Fetch job listings from Adzuna API"""
    # Base URL for the API - notice the country parameter
    base_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
    
    # Build query parameters
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "content-type": "application/json"
    }
    
    # Add optional parameters if provided - only add if not empty
    if what and what.strip():
        params["what"] = what.strip()
    if where and where.strip():
        params["where"] = where.strip()
    if category and category.strip():
        params["category"] = category.strip()
    
    try:
        logger.info(f"Sending API request: page={page}, what={what}, where={where}, category={category}")
        # Make the API request
        response = requests.get(base_url, params=params)
        
        # For debugging
        logger.debug(f"Request URL: {response.url}")
        
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse and return JSON response
        return response.json()
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching jobs: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Response content: {e.response.text[:1000]}")  # Print first 1000 chars of error
        return None

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

def insert_jobs_into_db(conn, jobs_data):
    """Insert job data into the job_postings table, mapping API fields to database columns"""
    if not jobs_data or "results" not in jobs_data:
        logger.warning("No jobs found or invalid response")
        return 0
    
    cursor = conn.cursor()
    jobs_inserted = 0
    jobs_skipped = 0
    
    try:
        for job in jobs_data['results']:
            # Map Adzuna API fields to your database columns
            job_data = {
                'job_id': job.get('id', ''),
                'job_title': job.get('title', ''),
                'url': job.get('redirect_url', ''),
                'company_name': job.get('company', {}).get('display_name', ''),
                'description': job.get('description', ''),
                'location_short': get_location_short(job),
                'location_long': job.get('location', {}).get('display_name', ''),
                'state_code': get_state_code(job),
                'latitude': job.get('latitude'),
                'longitude': job.get('longitude'),
                'date_posted': parse_date(job.get('created', '')),
                'job_category': determine_job_category(job.get('title', ''), job.get('description', ''))
            }
            
            # Check if the job already exists (by job_id or by URL as fallback)
            cursor.execute(
                "SELECT job_id FROM job_postings WHERE job_id = %s OR url = %s", 
                (job_data['job_id'], job_data['url'])
            )
            
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

def get_location_short(job):
    """Extract short location from job data"""
    location_areas = job.get('location', {}).get('area', [])
    if len(location_areas) >= 4:  # Format appears to be [Country, State, County, City]
        return location_areas[3]  # This should be the city
    return ''

def get_state_code(job):
    """Extract state code from job data"""
    location_areas = job.get('location', {}).get('area', [])
    if len(location_areas) >= 2:  # Format appears to be [Country, State, County, City]
        return location_areas[1]  # This should be the state
    return ''

def parse_date(date_str):
    """Convert API date string to database format"""
    if not date_str:
        return None
    
    try:
        # Format from API: "2025-04-02T14:03:44Z"
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except ValueError:
        return None

def fetch_and_store_jobs(what=None, where=None, category=None, max_pages=5, country="us"):
    """Fetch jobs from Adzuna API and store them in the database"""
    conn = connect_to_db()
    if not conn:
        logger.error("Failed to connect to the database")
        return 0
    
    try:
        total_jobs_inserted = 0
        
        for page in range(1, max_pages + 1):
            logger.info(f"Fetching page {page} for '{what}' in '{where}'...")
            jobs_data = fetch_adzuna_jobs(what, where, category, page, country=country)
            
            if not jobs_data or "results" not in jobs_data or len(jobs_data["results"]) == 0:
                logger.info(f"No more results found after page {page-1}")
                break
                
            jobs_inserted = insert_jobs_into_db(conn, jobs_data)
            total_jobs_inserted += jobs_inserted
            
            # Log more detailed information
            logger.info(f"Page {page}: Found {len(jobs_data['results'])} jobs, inserted {jobs_inserted}")
            
            # If we want to be respectful of the API rate limits
            if page < max_pages:
                time.sleep(2)  # 2-second delay between page requests
                
        logger.info(f"Total jobs inserted for '{what}' in '{where}': {total_jobs_inserted}")
        return total_jobs_inserted
    
    finally:
        conn.close()

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
            
        # Create index on URL to help prevent duplicates
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM pg_indexes 
                WHERE tablename = 'job_postings' AND indexname = 'idx_url'
            );
        """)
        
        url_index_exists = cursor.fetchone()[0]
        
        if not url_index_exists:
            logger.info("Creating index on URL column")
            cursor.execute("CREATE INDEX idx_url ON job_postings (url);")
            conn.commit()
            
        return True
    except psycopg2.Error as e:
        logger.error(f"Error setting up database: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def run_job_search():
    """Run job search for all predefined categories and locations"""
    logger.info("Starting scheduled job search...")
    total_jobs = 0
    
    # Set up the job categories
    job_categories = {
        "STEM": STEM_JOBS,
        "Research": RESEARCH_JOBS,
        "Healthcare": HEALTHCARE_JOBS
    }
    
    # Loop through each category and location
    for category_name, job_list in job_categories.items():
        for job_title in job_list:
            for location in LOCATIONS:
                try:
                    # Use the job category name as the 'category' in Adzuna (if applicable)
                    adzuna_category = None
                    if category_name.lower() == "healthcare":
                        adzuna_category = "healthcare-nursing-social-services"
                    elif category_name.lower() == "stem" and "engineer" in job_title.lower():
                        adzuna_category = "engineering"
                    elif category_name.lower() == "stem" and "data" in job_title.lower():
                        adzuna_category = "it-jobs"
                    
                    # Log search attempt
                    logger.info(f"Searching for '{job_title}' in '{location}' (category: {adzuna_category})")
                    
                    # Perform search with limited pages to respect API limits
                    jobs_added = fetch_and_store_jobs(
                        what=job_title,
                        where=location,
                        category=adzuna_category,
                        max_pages=2
                    )
                    
                    total_jobs += jobs_added
                    
                    # Sleep to prevent hitting API rate limits
                    time.sleep(5)  # 5-second delay between searches
                    
                except Exception as e:
                    logger.error(f"Error searching for '{job_title}' in '{location}': {e}")
    
    logger.info(f"Job search completed. Total new jobs added: {total_jobs}")
    return total_jobs

def schedule_jobs():
    """Schedule jobs to run at specified intervals"""
    # Ensure database is set up
    if not setup_database():
        logger.error("Failed to set up database. Exiting.")
        return
    
    # Run immediately on startup
    logger.info("Running initial job search")
    run_job_search()
    
    # Schedule to run daily at specific times
    schedule.every().day.at("01:00").do(run_job_search)  # Early morning
    schedule.every().day.at("13:00").do(run_job_search)  # Mid-day
    
    logger.info("Scheduled job search to run at 01:00 and 13:00 daily")
    
    # Run the scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    logger.info("Adzuna job fetching automation started")
    
    # Command-line usage still available
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        print("Adzuna Job Search and Database Storage (Manual Mode)")
        print("-" * 50)
        
        what = input("Enter job title or keywords (leave blank for all): ").strip()
        where = input("Enter location (leave blank for anywhere): ").strip()
        category = input("Enter category (it, engineering, healthcare, etc.) - leave blank for all: ").strip()
        
        try:
            max_pages_input = input("Enter maximum number of pages to fetch (default 5): ").strip()
            max_pages = int(max_pages_input) if max_pages_input else 5
        except ValueError:
            print(f"Invalid input for maximum pages. Using default value of 5.")
            max_pages = 5
        
        country_code = input("Enter country code (us, gb, au, ca, de, etc. - default is us): ").strip() or "us"
        
        fetch_and_store_jobs(what, where, category, max_pages, country_code)
    else:
        # Run automated mode
        schedule_jobs()