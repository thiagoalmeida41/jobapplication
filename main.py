import logging
import os
import json
from io import BytesIO
from typing import Optional, Dict

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
except ValueError as e:
    logger.critical(f"Configuration Error: {e}")
    raise RuntimeError(f"Gemini API configuration failed: {e}. The application cannot start.")
except Exception as e:
    logger.critical(f"An unexpected error occurred during Gemini API configuration: {e}", exc_info=True)
    raise RuntimeError(f"An unexpected failure occurred during Gemini API configuration: {e}")

# --- Initialize Gemini Model ---
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Successfully initialized the 'gemini-1.5-flash' generative model.")
except Exception as e:
    logger.critical(f"Failed to initialize the Gemini model: {e}", exc_info=True)
    raise RuntimeError(f"Could not initialize the generative model: {e}")

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="Job Application AI Assistant",
    description="An API that uses Gemini to analyze a job description and CV.",
    version="1.1.0"
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

async def safe_gemini_call(prompt: str, prompt_name: str, part_key: str, config: Dict) -> any:
    """A robust wrapper for making calls to the Gemini API."""
    logger.info(f"Sending prompt: '{prompt_name}'...")
    try:
        response = model.generate_content(prompt, generation_config=config)
        parsed_data = json.loads(response.text)
        result = parsed_data.get(part_key)
        if result is not None:
            logger.info(f"Prompt '{prompt_name}' successful.")
            return result
        else:
             logger.warning(f"'{prompt_name}' completed but key '{part_key}' not in response.")
             return None
    except Exception as e:
        logger.error(f"Error during '{prompt_name}' execution: {e}", exc_info=True)
        # Try to clean up the response text if it's a JSON error
        try:
            raw_text = response.text
            clean_text = raw_text.strip().replace("```json", "").replace("```", "")
            parsed_data = json.loads(clean_text)
            logger.warning(f"Successfully recovered from malformed JSON in '{prompt_name}'.")
            return parsed_data.get(part_key)
        except Exception:
            logger.error(f"Failed to recover from JSON error for '{prompt_name}'.")
            return None

async def analyze_job_posting_with_gemini(
    job_description_raw: str, job_url: str, cv_content: Optional[str]
) -> Dict[str, any]:
    """Orchestrates a series of prompts to Gemini AI for analysis."""
    if not job_description_raw or not job_description_raw.strip():
        raise ValueError("Job description content is empty.")

    json_config = {"response_mime_type": "application/json"}
    
    # --- Prompt 1: Core Information Extraction ---
    prompt1 = f"""Analyze the job posting to extract: JOB_TITLE, COMPANY, LOCATION, and a full JOB_DESCRIPTION. For JOB_DESCRIPTION, capture all relevant details. Exclude generic boilerplate. Format as JSON. If not found, use "N/A". Job Posting: --- {job_description_raw} ---"""
    logger.info("Sending prompt: 'Core Job Information Extraction'...")
    job_details = {"JOB_TITLE": "N/A", "COMPANY": "N/A", "LOCATION": "N/A", "JOB_DESCRIPTION": job_description_raw}
    try:
        response1 = model.generate_content(prompt1, generation_config=json_config)
        parsed_data = json.loads(response1.text)
        job_details = {
            "JOB_TITLE": parsed_data.get("JOB_TITLE", "N/A"),
            "COMPANY": parsed_data.get("COMPANY", "N/A"),
            "LOCATION": parsed_data.get("LOCATION", "N/A"),
            "JOB_DESCRIPTION": parsed_data.get("JOB_DESCRIPTION", job_description_raw)
        }
        logger.info(f"Core information extracted for: {job_details['JOB_TITLE']}")
    except Exception as e:
        logger.error(f"Error in Core Info prompt: {e}. Proceeding with raw data.", exc_info=True)

    cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n" if cv_content else ""
    
    # --- Subsequent Prompts ---
    prompt_challenge = f"Based on this job description for '{job_details['JOB_TITLE']}', what's the biggest challenge and its root cause? Job Description: --- {job_details['JOB_DESCRIPTION']} ---"
    challenge_and_root_cause = await safe_gemini_call(prompt_challenge, "Biggest Challenge", "CHALLENGE_AND_ROOT_CAUSE", json_config) or "N/A"

    prompt_hook = f"Write a cover letter hook (under 100 words) for '{job_details['JOB_TITLE']}'. Empathize with their challenge: '{challenge_and_root_cause}'. Use my CV to connect my experience. Job Description: --- {job_details['JOB_DESCRIPTION']} --- {cv_context}"
    cover_letter_hook = await safe_gemini_call(prompt_hook, "Cover Letter Hook", "COVER_LETTER_HOOK", json_config) or "N/A"

    prompt_cover_letter = f"Write a full cover letter for '{job_details['JOB_TITLE']}'. Start with this hook: '{cover_letter_hook}'. Expand on how my CV shows I can solve their challenges. Job Description: --- {job_details['JOB_DESCRIPTION']} --- {cv_context}"
    cover_letter = await safe_gemini_call(prompt_cover_letter, "Full Cover Letter", "COVER_LETTER", json_config) or "N/A"

    prompt_tell_me = f"Create a 'Tell me about yourself' pitch, aligning my CV with the needs of the '{job_details['JOB_TITLE']}' role. Job Description: --- {job_details['JOB_DESCRIPTION']} --- {cv_context}"
    tell_me_about_yourself = await safe_gemini_call(prompt_tell_me, "Tell Me About Yourself", "TELL_ME_ABOUT_YOURSELF", json_config) or "N/A"
    
    prompt_cv_changes = f"As a resume expert, suggest 3-5 key CV optimizations against this job description. For each, provide 'original_cv_text' and a 'proposed_update'. Job Description: --- {job_details['JOB_DESCRIPTION']} --- My CV: --- {cv_content if cv_content else 'No CV content provided.'} ---"
    main_changes_to_my_cv = await safe_gemini_call(prompt_cv_changes, "CV Changes", "MAIN_CHANGES_TO_MY_CV", json_config) or []
    
    prompt_questions = f"Suggest 3-5 insightful questions I should ask an interviewer for the '{job_details['JOB_TITLE']}' role. Job Description: --- {job_details['JOB_DESCRIPTION']} --- {cv_context}"
    questions_to_ask = await safe_gemini_call(prompt_questions, "Questions to Ask", "QUESTIONS_TO_ASK", json_config) or []

    # --- Construct Final Output ---
    final_result = {
        "JOB_TITLE": job_details["JOB_TITLE"], "COMPANY": job_details["COMPANY"],
        "URL": job_url, "LOCATION": job_details["LOCATION"],
        "JOB_DESCRIPTION": job_details["JOB_DESCRIPTION"],
        "CHALLENGE_AND_ROOT_CAUSE": challenge_and_root_cause,
        "COVER_LETTER_HOOK": cover_letter_hook,
        "COVER_LETTER": cover_letter,
        "TELL_ME_ABOUT_YOURSELF": tell_me_about_yourself,
        "MAIN_CHANGES_TO_MY_CV": main_changes_to_my_cv,
        "QUESTIONS_TO_ASK": questions_to_ask
    }
    logger.info("All prompts executed. Final structured output prepared.")
    return final_result

# --- API Endpoint Definition ---
@app.post("/analyze_job_with_ai/")
async def analyze_job_endpoint(
    job_url: str = Form(""),
    job_description_raw: str = Form(...),
    cv_file: Optional[UploadFile] = File(None)
):
    """Main API endpoint to receive job data and trigger AI analysis."""
    logger.info(f"Received request for analysis. Job URL: {job_url if job_url else 'Not provided'}")
    cv_content = None
    if cv_file and cv_file.filename:
        logger.info(f"Processing CV file: {cv_file.filename}")
        cv_content = parse_cv_from_upload(cv_file)
    
    if not job_description_raw or not job_description_raw.strip():
        raise HTTPException(status_code=400, detail="job_description_raw cannot be empty.")

    try:
        analysis_result = await analyze_job_posting_with_gemini(
            job_description_raw=job_description_raw, job_url=job_url, cv_content=cv_content
        )
        return JSONResponse(content=analysis_result)
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Job Application AI Assistant is running and ready."}

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

