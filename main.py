from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import tempfile
import os
import json # Import json module for handling JSON data

# --- IMPORTANT: INTEGRATE YOUR AI SCRIPT'S LOGIC HERE ---
#
# Placeholder for your actual AI processing function.
# This function will take the job URL as a string and the CV file as a FastAPI UploadFile.
# It should return a dictionary with the structured output.
#
# EXAMPLE:
# Let's assume your core AI logic is in a separate file, e.g., 'your_ai_processor.py'
# and it has a function like `run_analysis(job_url, cv_filepath)` that returns the dictionary.
# from your_ai_processor import run_analysis # <--- UNCOMMENT AND REPLACE WITH YOUR ACTUAL IMPORT

async def process_job_application_ai(job_url: str, cv_file: UploadFile):
    """
    This function contains your AI logic.
    It takes the job URL and the uploaded CV file.
    It should return the structured dictionary output.

    You MUST replace the dummy 'result' dictionary with the actual output
    from your AI script.
    """
    try:
        # Read CV content from the uploaded file.
        # cv_bytes will contain the binary content of the file (e.g., PDF, DOCX).
        cv_bytes = await cv_file.read()

        # Determine file extension to handle different CV types if necessary
        file_extension = os.path.splitext(cv_file.filename)[1].lower()
        print(f"Received CV Filename: {cv_file.filename}, Extension: {file_extension}")
        print(f"Received Job URL: {job_url}")

        # --- YOUR AI SCRIPT INTEGRATION PART ---
        #
        # Here's how you might pass the CV to your existing script:
        #
        # Option 1: If your AI script needs a file path, save to a temporary file.
        # This is often needed for libraries that process files from disk (e.g., pypdf, python-docx).
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_cv_file:
            tmp_cv_file.write(cv_bytes)
            cv_filepath = tmp_cv_file.name

        # NOW, CALL YOUR ACTUAL AI SCRIPT FUNCTION
        # Example (replace 'run_analysis' with your function and parameters):
        # actual_ai_output = run_analysis(job_url, cv_filepath)
        #
        # IMPORTANT: Ensure your `run_analysis` (or equivalent) function
        # is capable of parsing the CV file from `cv_filepath`.
        # You might need libraries like `pypdf` for PDFs or `python-docx` for DOCX.
        #
        # For now, we'll use a dummy result as a placeholder:
        result = {
            "JOB_TITLE": "Senior AI Engineer",
            "COMPANY": "Global Innovators",
            "URL": job_url,
            "LOCATION": "Remote - Global",
            "JOB_DESCRIPTION": "Developing cutting-edge AI solutions for enterprise clients, leading cross-functional teams.",
            "CHALLENGE_AND_ROOT_CAUSE": "Identified root cause of data pipeline bottlenecks (inefficient data loading) and implemented optimized ETL processes, reducing processing time by 40%.",
            "COVER_LETTER_HOOK": "With a proven track record in developing scalable AI solutions and a passion for optimizing data workflows, I was immediately drawn to the Senior AI Engineer role at Global Innovators.",
            "COVER_LETTER": (
                "Dear Hiring Manager,\n\nI am writing to express my profound interest in the Senior AI Engineer "
                "position at Global Innovators, as advertised on your careers page. My experience aligns "
                "perfectly with your requirements for leading AI development and optimizing complex data pipelines. "
                "In my previous role, I spearheaded the development of an NLP-driven customer support system, "
                "which improved response times by 30% and significantly enhanced customer satisfaction. "
                "I am particularly excited about Global Innovators' commitment to cutting-edge research "
                "and believe my skills in machine learning, deep learning, and robust system design would be "
                "a significant asset to your team. I look forward to discussing how my expertise can contribute "
                "to your innovative projects.\n\nSincerely,\nThiago"
            ),
            "TELL_ME_ABOUT_YOURSELF": (
                "I'm a seasoned AI professional with 8 years of experience specializing in natural language "
                "processing and scalable data solutions. My career has focused on transforming complex "
                "data challenges into actionable AI-driven products. For example, at my last company, "
                "I led a project to build an intelligent recommendation engine that increased user engagement "
                "by 25%. I thrive in environments where I can combine technical expertise with strategic thinking "
                "to deliver impactful AI solutions."
            ),
            "MAIN_CHANGES_TO_MY_CV": [
                {"section": "Summary", "change": "Emphasize leadership in AI product development."},
                {"section": "Experience (Previous Role)", "change": "Add a bullet: 'Spearheaded NLP-driven customer support system, reducing response times by 30%.'"},
                {"section": "Skills", "change": "Ensure 'FastAPI' and 'PyTorch' are prominently listed."}
            ],
            "QUESTIONS_TO_ASK": [
                "What are the immediate priorities for the Senior AI Engineer in the first 90 days?",
                "How does Global Innovators approach continuous learning and professional development for its AI team?",
                "Could you describe a typical project lifecycle for an AI initiative from conception to deployment?"
            ]
        }

        # Clean up the temporary file after processing
        if 'cv_filepath' in locals() and os.path.exists(cv_filepath):
            os.unlink(cv_filepath)
            print(f"Cleaned up temporary CV file: {cv_filepath}")

        # Ensure the output is JSON serializable. Lists of dicts are fine.
        return result

    except Exception as e:
        print(f"Error during AI processing: {e}")
        # Reraise as an HTTPException to return a proper HTTP error response
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# Create the FastAPI application instance
app = FastAPI(
    title="Job Application AI Assistant API",
    description="API for helping job seekers prepare for job applications by analyzing job descriptions and CVs.",
    version="1.0.0"
)

# Define the API endpoint that Google Apps Script will call
@app.post("/analyze_job_application")
async def analyze_job_application_endpoint(
    job_url: str = Form(...),  # Expects job_url as form data
    cv_file: UploadFile = File(...) # Expects cv_file as an uploaded file
):
    """
    Receives a job description URL and a candidate's CV, then
    processes them using an AI model to generate job application insights.
    """
    if not job_url:
        raise HTTPException(status_code=400, detail="Job URL is required.")
    if not cv_file.filename: # Check if a file was actually provided
         raise HTTPException(status_code=400, detail="CV file is required.")

    print(f"API endpoint received request for URL: {job_url} and CV: {cv_file.filename}")

    # Call your core AI processing function
    processed_output = await process_job_application_ai(job_url, cv_file)

    # Return the structured JSON response
    return JSONResponse(content=processed_output)

# Optional: A simple root endpoint to check if the API is running
@app.get("/")
async def read_root():
    """
    Health check endpoint. Returns a simple message to confirm the API is running.
    """
    return {"message": "Job Application AI Assistant API is running!"}

