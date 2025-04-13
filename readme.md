# Job Search System

A modular job search system that uses BERT embeddings and FAISS indexing to match resumes or text queries with job postings.

## Project Structure

```
job_search/
├── __init__.py         # Package initialization
├── config.py           # Configuration settings
├── db.py               # Database connection utilities
├── bert_model.py       # BERT model initialization
├── embedding.py        # Text embedding functions
├── resume_parser.py    # Resume text extraction
├── index_builder.py    # FAISS index builder
├── job_matcher.py      # Job matching functionality
└── main.py             # Main application entry point
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd job-search-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Database Setup

Ensure your PostgreSQL database is set up with the following schema:

```sql
CREATE TABLE job_postings (
    job_id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    company VARCHAR(255),
    description TEXT,
    location VARCHAR(255),
    salary_range VARCHAR(255),
    job_type VARCHAR(50),
    post_date DATE,
    embedding FLOAT[]
);
```

Update the database connection details in `job_search/config.py`.

## Usage

### 1. Building the Index

Before using the search functionality, build the FAISS index:

```
python -m job_search.index_builder
```

This will:
- Calculate embeddings for job descriptions that don't have them
- Build a FAISS index and store it in the database
- Create a mapping between vector positions and job IDs

### 2. Searching Jobs

#### Search using a resume:

```
python -m job_search.main resume path/to/resume.pdf --limit 10
```

#### Search using a text query:

```
python -m job_search.main text "Python developer with machine learning experience" --limit 10
```

#### Save results to a file:

```
python -m job_search.main text "data scientist" --output results.json
```

## Advanced Usage

The system is built modularly so you can use its components in your own applications:

```python
from job_search.job_matcher import search_jobs
from job_search.resume_parser import get_resume_text

# Search with a text query
results = search_jobs("Python developer with AWS experience", top_k=5)

# Extract text from a resume
resume_text = get_resume_text(pdf_path="resume.pdf")
results = search_jobs(resume_text, top_k=5)
```

## Performance Considerations

- The FAISS index uses IVFPQ for efficient similarity search
- Index building should be done periodically as new job postings are added
- For very large datasets, consider adjusting the number of clusters and nprobe value
