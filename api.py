import os
import json
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the core logic functions from your url_analyzer.py script
# Ensure url_analyzer.py is in the same directory or accessible in PYTHONPATH
from url_analyzer import scrape_url_content, parse_cv_document, analyze_job_posting_with_gemini

# Configure logging for the API application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Job Application AI Assistant API",
    description="API to analyze job postings and generate application content using Gemini AI.",
    version="1.0.0"
)

# --- Pydantic Model for Request Body (if using JSON for URL only) ---
class JobAnalysisRequest(BaseModel):
    url: str

# --- API Endpoint: Analyze Job Posting ---
@app.post("/analyze_job/")
async def analyze_job(
    url: str = Form(..., description="The URL of the job posting to analyze."),
    cv_file: Optional[UploadFile] = File(None, description="Optional: Your CV in .docx format.")
):
    """
    Analyzes a job posting from a given URL and optionally uses a provided CV
    to generate personalized application content.

    Parameters:
    - url: The URL of the job posting (e.g., https://www.linkedin.com/jobs/view/...).
    - cv_file: Your CV file in .docx format (optional).
    """
    logging.info(f"Received request to analyze job: {url}")
    cv_content = None

    # 1. Handle CV file upload
    if cv_file:
        if not cv_file.filename.lower().endswith('.docx'):
            logging.error(f"Unsupported CV file format: {cv_file.filename}. Only .docx is supported.")
            raise HTTPException(status_code=400, detail="Only .docx files are supported for CV.")

        # Save the uploaded file temporarily to process it
        temp_cv_path = f"/tmp/{cv_file.filename}" if os.name != 'nt' else f"C:\\Temp\\{cv_file.filename}"
        os.makedirs(os.path.dirname(temp_cv_path), exist_ok=True) # Ensure directory exists

        try:
            with open(temp_cv_path, "wb") as buffer:
                content = await cv_file.read()
                buffer.write(content)
            logging.info(f"CV file '{cv_file.filename}' saved temporarily to '{temp_cv_path}'.")

            # Parse CV content
            cv_content = parse_cv_document(temp_cv_path)
            if cv_content is None:
                logging.warning(f"Failed to parse CV from '{temp_cv_path}'. Analysis will proceed without CV content.")
                # If parsing fails, remove the temp file but allow process to continue
                os.remove(temp_cv_path)
                raise HTTPException(status_code=422, detail=f"Failed to parse CV content. Please check your .docx file. Analysis will continue without CV data.")

        except Exception as e:
            logging.error(f"Error processing CV file: {e}", exc_info=True)
            if os.path.exists(temp_cv_path):
                os.remove(temp_cv_path) # Clean up temp file on error
            raise HTTPException(status_code=500, detail=f"Internal server error processing CV: {str(e)}")
        finally:
            if os.path.exists(temp_cv_path):
                os.remove(temp_cv_path) # Ensure temp file is removed after use

    # 2. Scrape Job Posting Content
    scraped_content = scrape_url_content(url)

    if not scraped_content:
        logging.error(f"Failed to scrape content from URL: {url}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job posting content from the provided URL.")

    # 3. Analyze Content with Gemini AI
    try:
        analysis_result = analyze_job_posting_with_gemini(scraped_content, url, cv_content)
        return JSONResponse(content=analysis_result)
    except Exception as e:
        logging.error(f"Error during Gemini analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during AI analysis: {str(e)}")

# --- Run the application with Uvicorn (usually done via command line) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

