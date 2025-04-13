"""
AI Job Matcher - FastAPI Backend

This file contains the API endpoints for the AI Job Matcher application.
It handles resume uploads, text searches, and returns job matches.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import shutil
import sys

# Add the job_search package to path
sys.path.append('.')  # Adjust this path as needed for your environment

# Import job_search modules
from job_search.resume_parser import get_resume_text
from job_search.job_matcher import search_jobs

# Define response models
class JobResult(BaseModel):
    """Model for a job search result."""
    job_id: int
    title: str
    company: str
    location: Optional[str] = "Not specified"
    similarity_score: float
    job_type: Optional[str] = "Not specified"
    salary_range: Optional[str] = "Not specified"
    description: str
    description_preview: str
    url: Optional[str] = None

class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[JobResult]
    total: int
    page: int  
    total_pages: int
    query_type: str
    query_text: str

# Create FastAPI app
app = FastAPI(
    title="AI Job Matcher API",
    description="Match resumes with job postings using AI",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    """Root endpoint that returns API status."""
    return {"status": "AI Job Matcher API is running", "version": "1.0.0"}

@app.get("/search/text", response_model=SearchResponse)
async def search_by_text(query: str, limit: int = 10, page: int = 1):
    """
    Search for jobs using a text query with pagination.
    
    Args:
        query: Text query for job search
        limit: Maximum number of results to return per page
        page: Page number to return
        
    Returns:
        SearchResponse: Job search results with pagination
    """
    try:
        # Search for matching jobs using FAISS with pagination
        search_result = search_jobs(query, top_k=200, page=page, limit=limit)
        
        # Format results for response
        formatted_results = []
        for job in search_result["results"]:
            description = job.get("description", "")
            formatted_results.append(JobResult(
                job_id=job["job_id"],
                title=job["title"],
                company=job["company"],
                location=job.get("location_long", job.get("location_short", "Not specified")),
                similarity_score=job["similarity_score"],
                job_type=job.get("job_category", "Not specified"),  
                salary_range=job.get("salary_range", "Not specified"),
                description=description,
                description_preview=description[:200] + "..." if len(description) > 200 else description,
                url=job.get("url", None)  
            ))
        
        return SearchResponse(
            results=formatted_results,
            total=search_result["total"],
            page=search_result["page"],
            total_pages=search_result["total_pages"],
            query_type="text",
            query_text=query
        )
    except Exception as e:
        # Return error details to client
        raise HTTPException(status_code=500, detail=f"Error searching jobs: {str(e)}")

@app.post("/search/resume", response_model=SearchResponse)
async def search_by_resume(file: UploadFile = File(...), limit: int = Form(10), page: int = Form(1)):
    """
    Search for jobs by uploading a resume with pagination.
    
    Args:
        file: Uploaded resume file (PDF or DOCX)
        limit: Maximum number of results to return per page
        page: Page number to return
        
    Returns:
        SearchResponse: Job search results with pagination
    """
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Create temporary file to store the upload
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file to temp location
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Extract text based on file type
        if file_ext == '.pdf':
            resume_text = get_resume_text(pdf_path=temp_file_path)
        else:
            resume_text = get_resume_text(docx_path=temp_file_path)
        
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from resume")
        
        # Search for matching jobs with pagination
        search_result = search_jobs(resume_text, top_k=50, page=page, limit=limit)
        
        # Format results
        formatted_results = []
        for job in search_result["results"]:
            description = job.get("description", "")
            formatted_results.append(JobResult(
                job_id=job["job_id"],
                title=job["title"],
                company=job["company"],
                location=job.get("location_long", job.get("location_short", "Not specified")),
                similarity_score=job["similarity_score"],
                job_type=job.get("job_category", "Not specified"),
                salary_range=job.get("salary_range", "Not specified"),
                description=description,
                description_preview=description[:200] + "..." if len(description) > 200 else description,
                url=job.get("url", None)
            ))
        
        return SearchResponse(
            results=formatted_results,
            total=search_result["total"],
            page=search_result["page"],
            total_pages=search_result["total_pages"],
            query_type="resume",
            query_text=f"Resume: {file.filename}"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir)

# Run the application with uvicorn when script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)