import logging
import os
import json
from io import BytesIO
from typing import Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from docx import Document
from docx.opc.exceptions import OpcError
import google.generativeai as genai

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Gemini API Configuration ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("The GOOGLE_API_KEY environment variable is not set.")
    genai.configure(api_key=gemini_api_key)
    logger.info("Successfully configured the Gemini API client.")
except Exception as e:
    logger.critical(f"FATAL: Gemini API configuration failed: {e}", exc_info=True)
    raise RuntimeError(f"FATAL: Gemini API configuration failed: {e}") from e

# --- Initialize Gemini Model ---
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully initialized the 'gemini-1.5-flash' generative model.")
except Exception as e:
    logger.critical(f"FATAL: Could not initialize Gemini model: {e}", exc_info=True)
    raise RuntimeError(f"FATAL: Could not initialize Gemini model: {e}") from e

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Job Application AI Assistant",
    description="An API that uses Gemini to analyze a job description and CV.",
    version="1.4.0" # Incremented version to ensure change
)

# --- Core Functions ---

def parse_cv_from_upload(file: UploadFile) -> Optional[str]:
    """Parses an uploaded .docx file and extracts its text content."""
    if not file.filename.lower().endswith('.docx'):
        logger.warning(f"Unsupported file format for CV: '{file.filename}'.")
        return None
    try:
        file_stream = BytesIO(file.file.read())
        doc = Document(file_stream)
        full_text = [para.text for para in doc.paragraphs]
        cv_content = "\n".join(full_text).strip()
        logger.info(f"Successfully parsed CV from '{file.filename}'.")
        return cv_content
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing CV '{file.filename}': {e}", exc_info=True)
        return None

async def safe_gemini_call(prompt: str, prompt_name: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A robust wrapper for making calls to the Gemini API with a specific schema."""
    logger.info(f"Sending prompt: '{prompt_name}'...")
    config = {
        "response_mime_type": "application/json",
        "response_schema": schema
    }
    try:
        response = model.generate_content(prompt, generation_config=config)
        parsed_data = json.loads(response.text)
        logger.info(f"Prompt '{prompt_name}' successful.")
        return parsed_data
    except json.JSONDecodeError as json_err:
        logger.error(f"JSONDecodeError during '{prompt_name}': {json_err}. Raw response: '{getattr(response, 'text', 'N/A')}'")
        return None
    except Exception as e:
        logger.error(f"An unexpected error during '{prompt_name}' execution: {e}", exc_info=True)
        return None

async def analyze_job_posting_with_gemini(
    job_description_raw: str, job_url: str, cv_content: Optional[str]
) -> Dict[str, Any]:
    """Orchestrates a series of prompts to Gemini AI for analysis."""
    if not job_description_raw or not job_description_raw.strip():
        raise ValueError("Job description content is empty.")

    # --- Schema Definitions for each prompt ---
    info_schema = {"type": "OBJECT", "properties": {"JOB_TITLE": {"type": "STRING"}, "COMPANY": {"type": "STRING"}, "LOCATION": {"type": "STRING"}, "JOB_DESCRIPTION": {"type": "STRING"}}}
    challenge_schema = {"type": "OBJECT", "properties": {"CHALLENGE_AND_ROOT_CAUSE": {"type": "STRING"}}}
    hook_schema = {"type": "OBJECT", "properties": {"COVER_LETTER_HOOK": {"type": "STRING"}}}
    cover_letter_schema = {"type": "OBJECT", "properties": {"COVER_LETTER": {"type": "STRING"}}}
    tell_me_schema = {"type": "OBJECT", "properties": {"TELL_ME_ABOUT_YOURSELF": {"type": "STRING"}}}
    cv_changes_schema = {"type": "OBJECT", "properties": {"MAIN_CHANGES_TO_MY_CV": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"original_cv_text": {"type": "STRING"}, "proposed_update": {"type": "STRING"}}}}}}
    questions_schema = {"type": "OBJECT", "properties": {"QUESTIONS_TO_ASK": {"type": "ARRAY", "items": {"type": "STRING"}}}}

    # --- Prompt Execution ---
    prompt1 = f"""Analyze the job posting to extract: JOB_TITLE, COMPANY, LOCATION, and a full JOB_DESCRIPTION. For JOB_DESCRIPTION, capture all relevant details. Exclude generic boilerplate. Format as JSON. If not found, use "N/A". Job Posting: --- {job_description_raw} ---"""
    job_details_data = await safe_gemini_call(prompt1, "Core Job Information", info_schema)
    job_details = job_details_data if job_details_data else {"JOB_TITLE": "N/A", "COMPANY": "N/A", "LOCATION": "N/A", "JOB_DESCRIPTION": job_description_raw}
    
    cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n" if cv_content else ""
    
    prompt_challenge = f"Based on this job description for '{job_details.get('JOB_TITLE')}', what's the biggest challenge and its root cause? Respond with clean JSON, no invalid characters. Job Description: --- {job_details.get('JOB_DESCRIPTION')} ---"
    challenge_data = await safe_gemini_call(prompt_challenge, "Biggest Challenge", challenge_schema)
    
    prompt_hook = f"Write a cover letter hook (under 100 words) for '{job_details.get('JOB_TITLE')}'. Empathize with their challenge: '{challenge_data.get('CHALLENGE_AND_ROOT_CAUSE') if challenge_data else ''}'. Use my CV to connect my experience. Job Description: --- {job_details.get('JOB_DESCRIPTION')} --- {cv_context}"
    hook_data = await safe_gemini_call(prompt_hook, "Cover Letter Hook", hook_schema)

    prompt_cover_letter = f"Write a full cover letter for '{job_details.get('JOB_TITLE')}'. Start with this hook: '{hook_data.get('COVER_LETTER_HOOK') if hook_data else ''}'. Expand on how my CV shows I can solve their challenges. Job Description: --- {job_details.get('JOB_DESCRIPTION')} --- {cv_context}"
    cover_letter_data = await safe_gemini_call(prompt_cover_letter, "Full Cover Letter", cover_letter_schema)

    prompt_tell_me = f"Create a 'Tell me about yourself' pitch, aligning my CV with the needs of the '{job_details.get('JOB_TITLE')}' role. Job Description: --- {job_details.get('JOB_DESCRIPTION')} --- {cv_context}"
    tell_me_data = await safe_gemini_call(prompt_tell_me, "Tell Me About Yourself", tell_me_schema)
    
    prompt_cv_changes = f"As a resume expert, suggest 3-5 key CV optimizations against this job description. For each, provide 'original_cv_text' and a 'proposed_update'. Job Description: --- {job_details.get('JOB_DESCRIPTION')} --- My CV: --- {cv_content if cv_content else 'No CV content provided.'} ---"
    cv_changes_data = await safe_gemini_call(prompt_cv_changes, "CV Changes", cv_changes_schema)
    
    prompt_questions = f"Suggest 3-5 insightful questions I should ask an interviewer for the '{job_details.get('JOB_TITLE')}' role. Job Description: --- {job_details.get('JOB_DESCRIPTION')} --- {cv_context}"
    questions_data = await safe_gemini_call(prompt_questions, "Questions to Ask", questions_schema)

    # --- Construct Final Output ---
    final_result = {
        "JOB_TITLE": job_details.get("JOB_TITLE", "N/A"),
        "COMPANY": job_details.get("COMPANY", "N/A"),
        "URL": job_url,
        "LOCATION": job_details.get("LOCATION", "N/A"),
        "JOB_DESCRIPTION": job_details.get("JOB_DESCRIPTION", job_description_raw),
        "CHALLENGE_AND_ROOT_CAUSE": challenge_data.get("CHALLENGE_AND_ROOT_CAUSE", "N/A") if challenge_data else "N/A",
        "COVER_LETTER_HOOK": hook_data.get("COVER_LETTER_HOOK", "N/A") if hook_data else "N/A",
        "COVER_LETTER": cover_letter_data.get("COVER_LETTER", "N/A") if cover_letter_data else "N/A",
        "TELL_ME_ABOUT_YOURSELF": tell_me_data.get("TELL_ME_ABOUT_YOURSELF", "N/A") if tell_me_data else "N/A",
        "MAIN_CHANGES_TO_MY_CV": cv_changes_data.get("MAIN_CHANGES_TO_MY_CV", []) if cv_changes_data else [],
        "QUESTIONS_TO_ASK": questions_data.get("QUESTIONS_TO_ASK", []) if questions_data else []
    }
    logger.info("All prompts executed. Final structured output prepared.")
    return final_result

# --- API Endpoint Definition ---
@app.post("/analyze_job_with_ai")
async def analyze_job_endpoint(
    job_url: str = Form(""),
    job_description_raw: str = Form(...),
    cv_file: Optional[UploadFile] = File(None)
):
    """Main API endpoint to receive job data and trigger AI analysis."""
    # Added extra logging to confirm the function is being entered.
    logger.info("--- REQUEST RECEIVED AT /analyze_job_with_ai ENDPOINT ---")
    logger.info(f"Job URL: {job_url if job_url else 'Not provided'}")
    cv_content = None
    if cv_file and cv_file.filename:
        logger.info(f"Processing CV file: {cv_file.filename}")
        cv_content = parse_cv_from_upload(cv_file)
    
    if not job_description_raw or not job_description_raw.strip():
        logger.error("Validation failed: job_description_raw cannot be empty.")
        raise HTTPException(status_code=400, detail="job_description_raw cannot be empty.")

    try:
        analysis_result = await analyze_job_posting_with_gemini(
            job_description_raw=job_description_raw, job_url=job_url, cv_content=cv_content
        )
        logger.info("--- ANALYSIS COMPLETE, SENDING 200 OK RESPONSE ---")
        return JSONResponse(content=analysis_result)
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")

@app.get("/")
async def root():
    return {"message": "Job Application AI Assistant is running and ready."}

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

