"""
Main application entry point for job search system.
"""

import os
import argparse
import json
from .job_matcher import search_jobs
from .resume_parser import get_resume_text

def search_resume(resume_path, top_k=10):
    """Search for jobs matching a resume.
    
    Args:
        resume_path (str): Path to resume file
        top_k (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing job details sorted by relevance
    """
    # Check if resume file exists
    if not os.path.exists(resume_path):
        print(f"Error: Resume file '{resume_path}' not found!")
        return []
    
    # Determine file type based on extension
    file_ext = os.path.splitext(resume_path)[1].lower()
    
    # Extract text from the resume
    if file_ext == '.pdf':
        resume_text = get_resume_text(pdf_path=resume_path)
    elif file_ext == '.docx':
        resume_text = get_resume_text(docx_path=resume_path)
    else:
        print(f"Error: Unsupported file type '{file_ext}'. Only PDF and DOCX are supported.")
        return []
    
    if not resume_text:
        print("Error: Failed to extract text from the resume.")
        return []
    
    # Search for jobs matching the resume
    return search_jobs(resume_text, top_k)

def format_job_output(job):
    """Format job details for display.
    
    Args:
        job (dict): Job details
        
    Returns:
        dict: Formatted job details
    """
    return {
        "job_id": job["job_id"],
        "title": job["title"],
        "company": job["company"],
        "location": job.get("location", "Not specified"),
        "similarity_score": f"{job['similarity_score']:.2f}",
        "job_type": job.get("job_type", "Not specified"),
        "salary_range": job.get("salary_range", "Not specified"),
        "description_preview": job["description"][:200] + "..." if len(job["description"]) > 200 else job["description"]
    }

def display_results(results):
    """Display job search results to console.
    
    Args:
        results (list): List of job details
    """
    print(f"\nFound {len(results)} matching jobs:\n")
    for i, job in enumerate(results):
        print(f"{i+1}. {job['title']} at {job['company']} ({job['location']})")
        print(f"   Similarity Score: {job['similarity_score']}")
        print(f"   Job Type: {job['job_type']}")
        print(f"   Salary: {job['salary_range']}")
        print(f"   Description: {job['description_preview']}")
        print("")

def main():
    """Main function to run the job search application."""
    parser = argparse.ArgumentParser(description="Job Search Application")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Resume search command
    resume_parser = subparsers.add_parser("resume", help="Search jobs using a resume")
    resume_parser.add_argument("resume_path", help="Path to the resume file (PDF or DOCX)")
    resume_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    resume_parser.add_argument("--output", help="Output file for results (JSON format)")
    
    # Text search command
    text_parser = subparsers.add_parser("text", help="Search jobs using text query")
    text_parser.add_argument("query", help="Text query to search for")
    text_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")
    text_parser.add_argument("--output", help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    if args.command == "resume":
        results = search_resume(args.resume_path, args.limit)
    elif args.command == "text":
        results = search_jobs(args.query, args.limit)
    else:
        parser.print_help()
        return
    
    # Format output
    formatted_results = [format_job_output(job) for job in results]
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print results to console
        display_results(formatted_results)

if __name__ == "__main__":
    main()
