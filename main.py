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

    Args:
        job_description_raw: The raw text of the job description.
        job_url: The URL of the job posting.
        cv_content: The parsed text content of the user's CV.

    Returns:
        A dictionary containing the full analysis from the AI.
    """
    if not job_description_raw or not job_description_raw.strip():
        logger.error("Cannot analyze empty job description content.")
        raise ValueError("Job description content is empty or missing.")

    # This is the master JSON schema that the Google Apps Script expects.
    # All prompts will generate a part of this final structure.
    json_generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "JOB_TITLE": {"type": "STRING"},
                "COMPANY": {"type": "STRING"},
                "URL": {"type": "STRING"},
                "LOCATION": {"type": "STRING"},
                "JOB_DESCRIPTION": {"type": "STRING"},
                "CHALLENGE_AND_ROOT_CAUSE": {"type": "STRING"},
                "COVER_LETTER_HOOK": {"type": "STRING"},
                "COVER_LETTER": {"type": "STRING"},
                "TELL_ME_ABOUT_YOURSELF": {"type": "STRING"},
                "MAIN_CHANGES_TO_MY_CV": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "original_cv_text": {"type": "STRING"},
                            "proposed_update": {"type": "STRING"}
                        },
                        "required": ["original_cv_text", "proposed_update"]
                    }
                },
                "QUESTIONS_TO_ASK": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            }
        }
    }

    # --- Helper Function for Safe Gemini Calls ---
    async def safe_gemini_call(prompt: str, prompt_name: str, part_key: str) -> any:
        """A robust wrapper for making calls to the Gemini API."""
        logger.info(f"Sending prompt: '{prompt_name}'...")
        try:
            # Use async generation if available, otherwise fall back to sync
            # Note: The current google-generativeai library's primary `generate_content` is synchronous.
            # For a truly async implementation with FastAPI, `run_in_executor` would be used.
            # For simplicity and given the library's design, we call it directly.
            response = model.generate_content(
                prompt,
                generation_config=json_generation_config
            )
            parsed_data = json.loads(response.text)
            # The model is asked for the full schema but we extract just the part we need.
            result = parsed_data.get(part_key)
            if result is not None:
                logger.info(f"Prompt '{prompt_name}' successful.")
                return result
            else:
                 logger.warning(f"Prompt '{prompt_name}' completed but the key '{part_key}' was not found in the response.")
                 return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from '{prompt_name}': {e}. Raw response: '{response.text}'")
            return None
        except Exception as e:
            logger.error(f"Error during '{prompt_name}' execution: {e}", exc_info=True)
            return None


    # --- Prompt 1: Core Information Extraction ---
    # This prompt is designed to be robust and extract key details first.
    prompt1_extraction = f"""
    As an information extraction specialist, analyze the provided job posting text.
    Extract the JOB_TITLE, COMPANY, LOCATION, and a comprehensive JOB_DESCRIPTION.
    For JOB_DESCRIPTION, capture everything relevant to the role: responsibilities, all qualifications (required, preferred), benefits, and cultural points.
    Exclude generic company boilerplate, "how to apply" sections, and legal disclaimers at the very end.
    Format your output as a JSON object. If a field is not found, use "N/A".

    Job Posting Content:
    ---
    {job_description_raw}
    ---
    """
    logger.info("Sending prompt: 'Core Job Information Extraction'...")
    job_details = {"JOB_TITLE": "N/A", "COMPANY": "N/A", "LOCATION": "N/A", "JOB_DESCRIPTION": job_description_raw}
    try:
        response1 = model.generate_content(prompt1_extraction, generation_config=json_generation_config)
        parsed_data = json.loads(response1.text)
        job_details["JOB_TITLE"] = parsed_data.get("JOB_TITLE", "N/A")
        job_details["COMPANY"] = parsed_data.get("COMPANY", "N/A")
        job_details["LOCATION"] = parsed_data.get("LOCATION", "N/A")
        # Use the LLM-cleaned description if available, otherwise fall back to the raw one.
        job_details["JOB_DESCRIPTION"] = parsed_data.get("JOB_DESCRIPTION", job_description_raw)
        logger.info(f"Core information extracted. Job Title: {job_details['JOB_TITLE']}, Company: {job_details['COMPANY']}")
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error in 'Core Job Information Extraction' prompt: {e}. Proceeding with raw data.", exc_info=True)


    # Prepare context for subsequent prompts
    cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n" if cv_content else ""
    if not cv_context:
        logger.warning("No CV content provided. Subsequent analysis will be less personalized.")

    # --- Subsequent Prompts (Chained for better context) ---

    prompt_challenge = f"""
    Based on the job description for "{job_details['JOB_TITLE']}" at "{job_details['COMPANY']}", what is the biggest challenge for this role, and what is its likely root cause?
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    """
    challenge_and_root_cause = await safe_gemini_call(prompt_challenge, "Biggest Challenge", "CHALLENGE_AND_ROOT_CAUSE") or "N/A"

    prompt_hook = f"""
    Write an attention-grabbing cover letter hook (under 100 words) for the "{job_details['JOB_TITLE']}" role.
    Empathize with the identified challenge: "{challenge_and_root_cause}".
    Use my CV to connect my experience to solving this challenge.
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    {cv_context}
    """
    cover_letter_hook = await safe_gemini_call(prompt_hook, "Cover Letter Hook", "COVER_LETTER_HOOK") or "N/A"

    prompt_cover_letter = f"""
    Write a full, professional cover letter for the "{job_details['JOB_TITLE']}" role at "{job_details['COMPANY']}".
    Start with this hook: "{cover_letter_hook}".
    Then, expand on how my experience, detailed in my CV, makes me the perfect candidate to solve their challenges.
    Structure it with an intro, 2-3 body paragraphs linking my skills to the job's needs, and a strong closing.
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    {cv_context}
    """
    cover_letter = await safe_gemini_call(prompt_cover_letter, "Full Cover Letter", "COVER_LETTER") or "N/A"

    prompt_tell_me_about_yourself = f"""
    Create a compelling "Tell me about yourself" pitch (around 90 seconds).
    It should align my story from my CV with the needs of the "{job_details['JOB_TITLE']}" role.
    Structure it as: 1. Present Situation, 2. Past Experience highlights relevant to the job, 3. Future - why I'm excited about this specific opportunity.
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    {cv_context}
    """
    tell_me_about_yourself = await safe_gemini_call(prompt_tell_me_about_yourself, "Tell Me About Yourself", "TELL_ME_ABOUT_YOURSELF") or "N/A"

    prompt_cv_changes = f"""
    You are a resume expert. Analyze my CV against the job description.
    Suggest 3-5 key optimizations. For each, provide the 'original_cv_text' and a 'proposed_update' tailored to the job.
    If no CV is provided, give general advice with 'original_cv_text' as 'General Advice'.
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    My CV: --- {cv_content if cv_content else 'No CV content provided.'} ---
    """
    main_changes_to_my_cv = await safe_gemini_call(prompt_cv_changes, "CV Changes", "MAIN_CHANGES_TO_MY_CV") or []


    prompt_questions = f"""
    Suggest 3-5 insightful questions I should ask the interviewer for the "{job_details['JOB_TITLE']}" role.
    The questions should show strategic thinking and genuine interest in the company and team culture.
    Job Description: --- {job_details['JOB_DESCRIPTION']} ---
    {cv_context}
    """
    questions_to_ask = await safe_gemini_call(prompt_questions, "Questions to Ask", "QUESTIONS_TO_ASK") or []


    # --- Construct Final Structured Output ---
    final_result = {
        "JOB_TITLE": job_details["JOB_TITLE"],
        "COMPANY": job_details["COMPANY"],
        "URL": job_url,
        "LOCATION": job_details["LOCATION"],
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
    if cv_file:
        if cv_file.filename:
            logger.info(f"Processing uploaded CV file: {cv_file.filename}")
            cv_content = parse_cv_from_upload(cv_file)
            if not cv_content:
                logger.warning(f"Could not parse content from CV file: {cv_file.filename}. Proceeding without it.")
        else:
            logger.info("CV file was provided but has no filename. Skipping.")

    if not job_description_raw or not job_description_raw.strip():
        logger.error("Request rejected: job_description_raw is empty.")
        raise HTTPException(status_code=400, detail="The 'job_description_raw' field cannot be empty.")

    try:
        # This is the core logic call.
        analysis_result = await analyze_job_posting_with_gemini(
            job_description_raw=job_description_raw,
            job_url=job_url,
            cv_content=cv_content
        )
        # Return the complete analysis as a JSON response.
        return JSONResponse(content=analysis_result)

    except ValueError as ve:
        # Handle specific known errors, like empty content.
        logger.error(f"Validation error during analysis: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Handle unexpected errors during the Gemini analysis.
        logger.error(f"An unexpected error occurred during the analysis process: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while analyzing the job posting.")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Job Application AI Assistant is running."}

# --- Main Execution Block ---
# This allows you to run the server directly for local testing and development.
if __name__ == "__main__":
    # It's better practice to run uvicorn from the command line,
    # but this is convenient for development.
    # Example: uvicorn main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
