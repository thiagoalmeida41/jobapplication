import logging
import os
import json
from io import BytesIO
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from docx import Document
from docx.opc.exceptions import OpcError
import google.generativeai as genai

# --- Logging Configuration ---
# Set up a professional logging format.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Gemini API Configuration ---
# It's crucial to handle API key errors gracefully.
try:
    # Best practice: Load API key from environment variables for security.
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("The GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=gemini_api_key)
    logger.info("Successfully configured the Gemini API client.")
except ValueError as e:
    logger.critical(f"Configuration Error: {e}")
    # This is a fatal error, so we raise a RuntimeError to stop the application.
    raise RuntimeError(f"Gemini API configuration failed: {e}. The application cannot start.")
except Exception as e:
    logger.critical(f"An unexpected error occurred during Gemini API configuration: {e}", exc_info=True)
    raise RuntimeError(f"An unexpected failure occurred during Gemini API configuration: {e}")

# --- Initialize Gemini Model ---
# Initialize the model once on startup for efficiency.
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully initialized the 'gemini-1.5-flash' generative model.")
except Exception as e:
    logger.critical(f"Failed to initialize the Gemini model: {e}", exc_info=True)
    raise RuntimeError(f"Could not initialize the generative model: {e}")

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Job Application AI Assistant",
    description="An API that uses Gemini to analyze a job description and CV, generating tailored application materials.",
    version="1.0.0"
)

# --- Core Functions ---

def parse_cv_from_upload(file: UploadFile) -> Optional[str]:
    """
    Parses an uploaded .docx file and extracts its text content.
    This version reads the file from an in-memory stream.

    Args:
        file: An UploadFile object from FastAPI, expected to be a .docx file.

    Returns:
        A string containing the CV content, or None if parsing fails.
    """
    if not file.filename.lower().endswith('.docx'):
        logger.warning(f"Unsupported file format for CV: '{file.filename}'. Only .docx is supported.")
        return None
    try:
        # Read the file content into a BytesIO stream. python-docx can read from this file-like object.
        file_stream = BytesIO(file.file.read())
        doc = Document(file_stream)
        full_text = [para.text for para in doc.paragraphs]
        cv_content = "\n".join(full_text).strip()
        logger.info(f"Successfully parsed CV from uploaded file: '{file.filename}'.")
        return cv_content
    except OpcError as e:
        logger.error(f"Error reading .docx file '{file.filename}': {e}. It may be corrupted or not a valid format.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing CV '{file.filename}': {e}", exc_info=True)
        return None


async def analyze_job_posting_with_gemini(
    job_description_raw: str,
    job_url: str,
    cv_content: Optional[str]
) -> Dict[str, any]:
    """
    The core function that orchestrates a series of prompts to Gemini AI
    to analyze the job and CV, returning a structured dictionary of insights.
    """
    if not job_description_raw or not job_description_raw.strip():
        logger.error("Cannot analyze empty job description content.")
        raise ValueError("Job description content is empty or missing.")

    # Master JSON schema for the final output.
    json_generation_config = {"response_mime_type": "application/json"}

    # --- Prompt 1: Core Information Extraction ---
    prompt1_extraction = f"""
    Analyze the job posting to extract: JOB_TITLE, COMPANY, LOCATION, and a full JOB_DESCRIPTION.
    For JOB_DESCRIPTION, capture all relevant details: responsibilities, all qualifications, benefits, and cultural points.
    Exclude generic boilerplate and "how to apply" sections.
    Format output as JSON with these keys. If a field is not found, use "N/A".

    Job Posting Content: --- {job_description_raw} ---
    """
    logger.info("Sending prompt: 'Core Job Information Extraction'...")
    job_details = {"JOB_TITLE": "N/A", "COMPANY": "N/A", "LOCATION": "N/A", "JOB_DESCRIPTION": job_description_raw}
    try:
        response1 = model.generate_content(prompt1_extraction, generation_config=json_generation_config)
        parsed_data = json.loads(response1.text)
        job_details["JOB_TITLE"] = parsed_data.get("JOB_TITLE", "N/A")
        job_details["COMPANY"] = parsed_data.get("COMPANY", "N/A")
        job_details["LOCATION"] = parsed_data.get("LOCATION", "N/A")
        job_details["JOB_DESCRIPTION"] = parsed_data.get("JOB_DESCRIPTION", job_description_raw)
        logger.info(f"Core information extracted for: {job_details['JOB_TITLE']}")
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error in 'Core Job Information Extraction' prompt: {e}. Proceeding with raw data.", exc_info=True)

    # Prepare context for subsequent prompts
    cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n" if cv_content else ""

    # Subsequent prompts using the extracted details...
    # (The logic for the other prompts remains the same, so it is omitted here for brevity
    # but should be included in your file. The important part is the endpoint definition below.)
    challenge_and_root_cause = "Generated via prompt..."
    cover_letter_hook = "Generated via prompt..."
    cover_letter = "Generated via prompt..."
    tell_me_about_yourself = "Generated via prompt..."
    main_changes_to_my_cv = []
    questions_to_ask = []


    # --- Construct Final Structured Output ---
    final_result = {
        "JOB_TITLE": job_details["JOB_TITLE"], "COMPANY": job_details["COMPANY"],
        "URL": job_url, "LOCATION": job_details["LOCATION"],
        "JOB_DESCRIPTION": job_details["JOB_DESCRIPTION"],
        "CHALLENGE_AND_ROOT_CAUSE": challenge_and_root_cause, # Placeholder
        "COVER_LETTER_HOOK": cover_letter_hook, # Placeholder
        "COVER_LETTER": cover_letter, # Placeholder
        "TELL_ME_ABOUT_YOURSELF": tell_me_about_yourself, # Placeholder
        "MAIN_CHANGES_TO_MY_CV": main_changes_to_my_cv, # Placeholder
        "QUESTIONS_TO_ASK": questions_to_ask # Placeholder
    }
    logger.info("All prompts executed. Final structured output prepared.")
    return final_result

# --- API Endpoint Definition ---
# This is the critical part that defines the URL path.
@app.post("/analyze_job_with_ai/")
async def analyze_job_endpoint(
    job_url: str = Form(...),
    job_description_raw: str = Form(...),
    cv_file: Optional[UploadFile] = File(None)
):
    """
    This is the main API endpoint that the Google Apps Script will call.
    It receives the job data, parses the CV, and triggers the AI analysis.
    """
    logger.info(f"Received request for job URL: {job_url}")
    cv_content = None
    if cv_file and cv_file.filename:
        logger.info(f"Processing uploaded CV file: {cv_file.filename}")
        cv_content = parse_cv_from_upload(cv_file)
    
    if not job_description_raw or not job_description_raw.strip():
        logger.error("Request rejected: job_description_raw is empty.")
        raise HTTPException(status_code=400, detail="The 'job_description_raw' field cannot be empty.")

    try:
        analysis_result = await analyze_job_posting_with_gemini(
            job_description_raw=job_description_raw, job_url=job_url, cv_content=cv_content
        )
        return JSONResponse(content=analysis_result)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the analysis process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during analysis.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Job Application AI Assistant is running and ready for analysis."}

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
