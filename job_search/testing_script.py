"""
Test script to search for jobs using a resume.
"""

import os
# Set environment variable to handle OpenMP conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import necessary modules
import sys
import traceback

# Add the parent directory to sys.path if needed
sys.path.insert(0, '.')  # if you're running from the project root
# sys.path.insert(0, 'D:/job-search-system')  # or specify the full path

try:
    # Import from the job_search package
    from job_search.resume_parser import get_resume_text
    from job_search.job_matcher import search_jobs, get_job_columns
    
    # First check what columns are in your database
    print("Checking database columns:")
    columns = get_job_columns()
    
    # Either use a resume file
    resume_path = r"C:\Users\Omen\Downloads\resume.pdf"  # Update this path
    if os.path.exists(resume_path):
        print(f"Extracting text from resume: {resume_path}")
        resume_text = get_resume_text(pdf_path=resume_path)
        if resume_text:
            print(f"Successfully extracted {len(resume_text)} characters from resume")
            query = resume_text
        else:
            print("Failed to extract text from resume")
            query = "Python developer with machine learning experience"  # Fallback query
    else:
        # Or use a text query directly
        print("No resume file found, using text query")
        query = "Python developer with machine learning experience"
    
    print(f"Searching with query: {query[:100]}...")
    
    # Search for matching jobs
    matches = search_jobs(query, top_k=5)
    
    # Display results
    print("\nSearch Results:")
    if matches:
        for i, job in enumerate(matches):
            print(f"\n{i+1}. Job ID: {job['job_id']}")
            
            # Print all available fields
            for field, value in job.items():
                if field != 'description':  # Skip long description for clarity
                    print(f"   {field}: {value}")
            
            # Print shortened description
            if 'description' in job:
                desc = job['description']
                print(f"   Description preview: {desc[:200]}..." if len(desc) > 200 else desc)
    else:
        print("No matching jobs found")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
