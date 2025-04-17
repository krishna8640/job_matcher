import logging
import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import shutil

# Ensure job_search is on PYTHONPATH (for local development)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Application logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import job_search modules
from job_search.resume_parser import get_resume_text
from job_search.job_matcher import search_jobs
from job_search.index_cache import IndexCache

class JobResult(BaseModel):
    """Model for a job search result."""
    job_id: str = Field(..., description="Job identifier as text")
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

app = FastAPI(
    title="AI Job Matcher API",
    description="Match resumes with job postings using AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize resources at application startup."""
    logger.info("Pre-loading FAISS index...")
    IndexCache.get_instance().load_index()
    logger.info("Startup initialization complete.")

@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint."""
    return {"status": "AI Job Matcher API is running", "version": app.version}

@app.get("/search/text", response_model=SearchResponse, tags=["Search"])
async def search_by_text(
    query: str,
    limit: int = 10,
    page: int = 1
):
    """Search for jobs using a text query with pagination."""
    try:
        logger.info(f"Text search: '{query}', page={page}, limit={limit}")
        search_result = search_jobs(query, top_k=200, page=page, limit=limit)

        formatted = []
        for job in search_result["results"]:
            desc = job.get("description", "")
            formatted.append(JobResult(
                job_id=str(job.get("job_id")),
                title=job.get("title", ""),
                company=job.get("company", ""),
                location=job.get("location", "Not specified"),
                similarity_score=job.get("similarity_score", 0.0),
                job_type=job.get("job_type", "Not specified"),
                salary_range=job.get("salary_range", "Not specified"),
                description=desc,
                description_preview=(desc[:200] + "...") if len(desc) > 200 else desc,
                url=job.get("url")
            ))

        return SearchResponse(
            results=formatted,
            total=search_result.get("total", 0),
            page=search_result.get("page", page),
            total_pages=search_result.get("total_pages", 1),
            query_type="text",
            query_text=query
        )
    except Exception:
        logger.exception("Error during text search")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search/resume", response_model=SearchResponse, tags=["Search"])
async def search_by_resume(
    file: UploadFile = File(...),
    limit: int = Form(10),
    page: int = Form(1)
):
    """Search for jobs by uploading a resume with pagination."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = (
            get_resume_text(pdf_path=temp_path)
            if file_ext == '.pdf' else
            get_resume_text(docx_path=temp_path)
        )
        if not resume_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from resume.")

        logger.info(f"Resume search: {file.filename}, page={page}, limit={limit}")
        search_result = search_jobs(resume_text, top_k=50, page=page, limit=limit)

        formatted = []
        for job in search_result["results"]:
            desc = job.get("description", "")
            formatted.append(JobResult(
                job_id=str(job.get("job_id")),
                title=job.get("title", ""),
                company=job.get("company", ""),
                location=job.get("location", "Not specified"),
                similarity_score=job.get("similarity_score", 0.0),
                job_type=job.get("job_type", "Not specified"),
                salary_range=job.get("salary_range", "Not specified"),
                description=desc,
                description_preview=(desc[:200] + "...") if len(desc) > 200 else desc,
                url=job.get("url")
            ))

        return SearchResponse(
            results=formatted,
            total=search_result.get("total", 0),
            page=search_result.get("page", page),
            total_pages=search_result.get("total_pages", 1),
            query_type="resume",
            query_text=file.filename
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error during resume search")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
