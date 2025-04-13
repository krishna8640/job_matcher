"""
Setup script for the job_search package.
"""

from setuptools import setup, find_packages

setup(
    name="job_search",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "faiss-cpu",
        "psycopg2-binary",
        "pdfplumber",
        "docx2txt",
    ],
    author="krishna korimerla",
    author_email="krishnakorimerla@gmail.com",
    description="Job search system using FAISS and all-mpnet-base-v2 embeddings",
    keywords="job search, faiss, all-mpnet-base-v2, embeddings",
    python_requires=">=3.6",
)
