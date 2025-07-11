import logging
import os
import json
from docx import Document
from docx.opc.exceptions import OpcError
from typing import Optional, List, Dict

import google.generativeai as genai

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini API Configuration ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
    logging.info("Gemini API key loaded from environment variable.")
except ValueError as e:
    logging.error(f"Configuration error: {e}")
    raise RuntimeError(f"Gemini API configuration failed: {e}. Please ensure GOOGLE_API_KEY is set.")
except Exception as e:
    logging.error(f"An unexpected error occurred during Gemini API configuration: {e}", exc_info=True)
    raise RuntimeError(f"Gemini API configuration failed unexpectedly: {e}")


# Initialize the Gemini model globally
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logging.error(f"Failed to initialize Gemini model 'gemini-1.5-flash': {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")


# --- Function: Parse CV Document (Synchronous) ---
def parse_cv_document(file_path: str) -> Optional[str]:
    """
    Parses a .docx file and extracts its text content.
    """
    if not os.path.exists(file_path):
        logging.error(f"CV file not found at: {file_path}")
        return None

    if not file_path.lower().endswith('.docx'):
        logging.warning(f"Unsupported file format for CV: {os.path.basename(file_path)}. "
                         "Only .docx files are directly supported by this parser. "
                         "Please convert your CV to .docx format or provide a .docx file.")
        return None

    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        cv_content = "\n".join(full_text).strip()
        logging.info(f"Successfully parsed CV from {file_path}. Length: {len(cv_content)} characters.")
        return cv_content
    except OpcError as e:
        logging.error(f"Error opening or reading .docx file {file_path}: {e}. "
                      "It might be corrupted or not a valid .docx file.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while parsing CV {file_path}: {e}", exc_info=True)
        return None


# --- Gemini Prompting Functions (Asynchronous, but doesn't call Playwright anymore) ---
async def analyze_job_posting_with_gemini(content: str, url: str, cv_content: Optional[str]) -> Dict[str, any]:
    """
    Analyzes job posting and CV content using Gemini AI.
    Returns a structured dictionary of insights.
    """
    final_result = {
        "JOB_TITLE": "N/A", "COMPANY": "N/A", "URL": url, "LOCATION": "N/A",
        "JOB_DESCRIPTION": "N/A", "CHALLENGE_AND_ROOT_CAUSE": "N/A",
        "COVER_LETTER_HOOK": "N/A", "COVER_LETTER": "N/A",
        "TELL_ME_ABOUT_YOURSELF": "N/A", "MAIN_CHANGES_TO_MY_CV": [],
        "QUESTIONS_TO_ASK": []
    }

    if not content:
        logging.error("Cannot analyze empty content for job posting.")
        final_result["JOB_DESCRIPTION"] = "Error: No content to analyze for job posting."
        final_result["COVER_LETTER"] = "N/A (No content)"
        final_result["TELL_ME_ABOUT_YOURSELF"] = "N/A (No content)"
        final_result["MAIN_CHANGES_TO_MY_CV"] = [{"section": "Error", "change": "No job description content."}]
        final_result["QUESTIONS_TO_ASK"] = ["N/A (No content)"]
        return final_result

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
            },
            "required": [
                "JOB_TITLE", "COMPANY", "URL", "LOCATION", "JOB_DESCRIPTION",
                "CHALLENGE_AND_ROOT_CAUSE", "COVER_LETTER_HOOK", "COVER_LETTER",
                "TELL_ME_ABOUT_YOURSELF", "MAIN_CHANGES_TO_MY_CV", "QUESTIONS_TO_ASK"
            ]
        }
    }


    prompt1_extraction = f"""
    As a highly skilled information extraction specialist for job postings, your task is to analyze the provided job offer text.

    Extract the "JOB_TITLE", "COMPANY" name, "LOCATION" (including "Remoto" if specified), and the "JOB_DESCRIPTION".

    For "JOB_DESCRIPTION", you must extract the **entire, comprehensive block of text** that details the job role, its requirements, and *all aspects of the offer to the candidate*. This includes:
    -   **Introduction to the role/opportunity**
    -   **All responsibilities and duties**
    -   **All required and preferred skills, qualifications, and experience** (e.g., "Diferenciais", "Nice to have", "Bonus Points")
    -   **Any specific details about the team, work environment, or culture directly associated with the role.**
    -   **All listed benefits, perks, and compensation details** (e.g., "What We Offer", "Benefits", "Paid Time Off", "Flexible Work", "Career Growth", "Culture" if describing employee experience).
    -   **Diversity, Equity, and Inclusion (DEI) statements** if they are an integral part of the job offer presentation.

    **Start extracting** the "JOB_DESCRIPTION" from the first clear heading or paragraph that introduces the job opportunity or the role's responsibilities (e.g., 'THE OPPORTUNITY', 'YOUR IMPACT', 'About the Role', 'What You'll Do', 'Responsibilities', 'Who You Are', 'What We Offer You').

    **Continue extracting ALL text that is part of the core job offer to the candidate.**

    **Stop extracting IMMEDIATELY** when you encounter sections that are clearly:
    -   General company boilerplate / "About Us" unrelated to the specific role's offer.
    -   Instructions on "how to apply" or application process details.
    -   Legal disclaimers (e.g., "Equal Opportunity Employer" if standalone and at the very end).
    -   Specific privacy policy notices or links (e.g., "Questions on how we treat your personal data? See our Aviso de Privacidade").
    -   Social media links or calls to action to follow the company.

    Ensure the extracted "JOB_DESCRIPTION" is a continuous block of text, preserving bullet points, paragraphs, and original formatting (like bolding or lists) where helpful for readability. The length of this description should be comprehensive, not a summary.

    Format your output as a JSON object with the specified keys. If a piece of information is not explicitly found, use "N/A" for its value.

    Job Posting Content:
    ---
    {content}
    ---
    """
    logging.info("Sending Prompt 1: Core Job Information Extraction (REVISED FOR BROADER JOB_DESCRIPTION)...")
    job_title, company, location, job_description_extracted = "N/A", "N/A", "N/A", "N/A"
    try:
        response1 = model.generate_content(
            prompt1_extraction,
            generation_config=json_generation_config # Use the full schema
        )
        parsed_data = json.loads(response1.text)
        job_title = parsed_data.get("JOB_TITLE", "N/A")
        company = parsed_data.get("COMPANY", "N/A")
        location = parsed_data.get("LOCATION", "N/A")
        job_description_extracted = parsed_data.get("JOB_DESCRIPTION", "N/A")
        logging.info(f"Prompt 1 successful. Job Title: {job_title}, Company: {company}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 1: {e}. Raw response: '{response1.text}'")
        cleaned_text = response1.text.strip()
        if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```json"): -len("```")].strip()
            try:
                parsed_data = json.loads(cleaned_text)
                job_title = parsed_data.get("JOB_TITLE", "N/A")
                company = parsed_data.get("COMPANY", "N/A")
                location = parsed_data.get("LOCATION", "N/A")
                job_description_extracted = parsed_data.get("JOB_DESCRIPTION", "N/A")
                logging.warning("Recovered from Markdown wrapper in Prompt 1 response.")
            except json.JSONDecodeError:
                logging.error("Failed to recover from Markdown wrapper.")
                pass
        logging.error("Failed to extract core job details, proceeding with N/A values.")
    except Exception as e:
        logging.error(f"Error during Prompt 1 execution: {e}", exc_info=True)
        logging.error("Failed to extract core job details, proceeding with N/A values.")

    job_description_for_prompts = job_description_extracted if job_description_extracted != "N/A" and job_description_extracted.strip() else content


    if job_description_for_prompts == "N/A" or not job_description_for_prompts.strip():
        logging.warning("Job Description is not available or empty. Skipping generation prompts.")
        return {
            "JOB_TITLE": job_title,
            "COMPANY": company,
            "URL": url, # Changed: URL from function argument
            "LOCATION": location,
            "JOB_DESCRIPTION": job_description_extracted, # Use extracted description
            "CHALLENGE_AND_ROOT_CAUSE": "N/A (Job Description not found)",
            "COVER_LETTER_HOOK": "N/A (Job Description not found)",
            "COVER_LETTER": "N/A (Job Description not found)",
            "TELL_ME_ABOUT_YOURSELF": "N/A (Job Description not found)",
            "MAIN_CHANGES_TO_MY_CV": [], # Changed: Default to empty list
            "QUESTIONS_TO_ASK": [] # Changed: Default to empty list
        }

    cv_context = ""
    if cv_content:
        cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n"
        logging.info("CV content provided for generation prompts.")
    else:
        logging.warning("No CV content provided or parsed for generation prompts. Results may be less personalized.")


    challenge_json_schema = {
        "type": "OBJECT",
        "properties": {
            "CHALLENGE_AND_ROOT_CAUSE": {"type": "STRING"}
        }
    }
    prompt_challenge_and_root_cause = f"""
    Based solely on the job description provided, what is the biggest challenge someone in this "{job_title}" position at "{company}" would face day-to-day?
    After identifying the challenge, give me the root cause of this specific issue.

    Respond strictly in JSON format, conforming to the main schema's "CHALLENGE_AND_ROOT_CAUSE" property.

    Job Title: {job_title}
    Company: {company}
    Job Description:
    ---
    {job_description_for_prompts}
    ---
    """
    logging.info("Sending NEW Prompt: Identify Biggest Challenge and Root Cause...")
    challenge_and_root_cause = "N/A"
    try:
        response_challenge = model.generate_content(
            prompt_challenge_and_root_cause,
            generation_config=json_generation_config # Use the full schema
        )
        parsed_data = json.loads(response_challenge.text)
        challenge_and_root_cause = parsed_data.get("CHALLENGE_AND_ROOT_CAUSE", "N/A")
        logging.info("NEW Prompt successful. Challenge and Root Cause identified.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Challenge Prompt: {e}. Raw response: '{response_challenge.text}'")
    except Exception as e:
        logging.error(f"Error during Prompt 3 execution: {e}", exc_info=True)


    hook_json_schema = {
        "type": "OBJECT",
        "properties": {
            "COVER_LETTER_HOOK": {"type": "STRING"}
        }
    }
    prompt_cover_letter_hook = f"""
    You're applying for this "{job_title}" position at "{company}".

    Write an attention-grabbing hook for your cover letter that highlights your experience based on the CV provided and qualifications in a way that shows you empathize and can successfully take on the challenges of the "{job_title}" role.
    Consider incorporating specific examples of how you've tackled these challenges in your past work, and explore creative ways to express your enthusiasm for the opportunity.
    Keep your hook within 100 words.

    Use the following identified challenge and root cause:
    Challenge and Root Cause: {challenge_and_root_cause}

    {cv_context}
    Respond strictly in JSON format, conforming to the main schema's "COVER_LETTER_HOOK" property.

    Job Title: {job_title}
    Company: {company}
    Job Description:
    ---
    {job_description_for_prompts}
    ---
    """
    logging.info("Sending NEW Prompt: Generate Cover Letter Hook...")
    cover_letter_hook = "N/A"
    try:
        response_hook = model.generate_content(
            prompt_cover_letter_hook,
            generation_config=json_generation_config # Use the full schema
        )
        parsed_data = json.loads(response_hook.text)
        cover_letter_hook = parsed_data.get("COVER_LETTER_HOOK", "N/A")
        logging.info("NEW Prompt successful. Cover Letter Hook generated.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Hook Prompt: {e}. Raw response: '{response_hook.text}'")
    except Exception as e:
        logging.error(f"Error during Prompt 3 execution: {e}", exc_info=True)


    # --- Prompt 4: Suggest Main Changes to CV (UPDATED for Two Columns) ---
    # Changed: Removed individual schema as full schema is in json_generation_config
    prompt4_cv_changes = f"""
    You are a resume expert. Based on the following job description AND the provided CV content, identify 3-5 key areas where an applicant's CV could be optimized to better align with this specific role.

    For each recommendation, you must provide it in two parts:
    1.  The **exact original phrase or sentence** (or a very short bullet point) from the CV that this recommendation pertains to.
    2.  A **proposed updated text** for that original phrase or sentence, tailored to highlight relevance for the job description.

    If the provided CV content is "No CV content provided.", or if no specific phrases can be identified for direct rephrasing, then provide 3-5 general recommendations. In this fallback case, for each recommendation, set "original_cv_text" to "General advice (No CV provided)" and provide a general proposed update in "proposed_update".

    Respond strictly in JSON format, conforming to the main schema's "MAIN_CHANGES_TO_MY_CV" property.

    Job Title: {job_title}
    Company: {company}
    Job Description:
    ---
    {job_description_for_prompts}
    ---
    My CV Content:
    ---
    {cv_content if cv_content else 'No CV content provided.'}
    ---
    """
    logging.info("Sending Prompt 4: Main Changes to CV (UPDATED for Two Columns)...")
    main_changes_to_my_cv = []
    try:
        response4 = model.generate_content(
            prompt4_cv_changes,
            generation_config=json_generation_config # Use the full schema
        )
        parsed_data = json.loads(response4.text)

        if isinstance(parsed_data.get("MAIN_CHANGES_TO_MY_CV"), list):
            temp_list = []
            for item in parsed_data["MAIN_CHANGES_TO_MY_CV"]:
                if isinstance(item, dict) and "original_cv_text" in item and "proposed_update" in item:
                    temp_list.append(item)
                else:
                    logging.warning(f"Invalid item format in MAIN_CHANGES_TO_MY_CV: {item}. Skipping.")
            main_changes_to_my_cv = temp_list
        else:
            logging.warning(f"MAIN_CHANGES_TO_MY_CV is not a list in parsed data: {parsed_data.get('MAIN_CHANGES_TO_MY_CV')}. Resetting to empty.")
            main_changes_to_my_cv = []

        logging.info("Prompt 4 successful. CV changes suggested.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 4: {e}. Raw response: '{response4.text}'")
        cleaned_text = response4.text.strip()
        if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```json"): -len("```")].strip()
            try:
                parsed_data = json.loads(cleaned_text)
                if isinstance(parsed_data.get("MAIN_CHANGES_TO_MY_CV"), list):
                    temp_list = []
                    for item in parsed_data["MAIN_CHANGES_TO_MY_CV"]:
                        if isinstance(item, dict) and "original_cv_text" in item and "proposed_update" in item:
                            temp_list.append(item)
                    main_changes_to_my_cv = temp_list
                    logging.warning("Recovered from Markdown wrapper in Prompt 4 response.")
                else:
                    main_changes_to_my_cv = []
            except json.JSONDecodeError:
                logging.error("Failed to recover from Markdown wrapper after JSONDecodeError.")
                main_changes_to_my_cv = []
        else:
            main_changes_to_my_cv = []
    except Exception as e:
        logging.error(f"Error during Prompt 4 execution: {e}", exc_info=True)
        main_changes_to_my_cv = []

    # --- Prompt 5: Generate Questions to Ask in Interview ---
    # Changed: Removed individual schema as full schema is in json_generation_config
    prompt5_questions_to_ask = f"""
    You are an interview coach. Based on the following job description and company, suggest 3-5 insightful questions a candidate should ask the interviewer.
    Focus on questions that show genuine interest, strategic thinking, or a desire to understand the team/company culture.
    {cv_context}
    Respond strictly in JSON format, conforming to the main schema's "QUESTIONS_TO_ASK" property.

    Job Title: {job_title}
    Company: {company}
    Job Description:
    ---
    {job_description_for_prompts}
    ---
    """
    logging.info("Sending Prompt 5: Questions to Ask...")
    questions_to_ask = []
    try:
        response5 = model.generate_content(
            prompt5_questions_to_ask,
            generation_config=json_generation_config # Use the full schema
        )
        parsed_data = json.loads(response5.text)
        questions_to_ask = parsed_data.get("QUESTIONS_TO_ASK", [])
        logging.info("Prompt 5 successful. Interview questions generated.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 5: {e}. Raw response: '{response5.text}'")
    except Exception as e:
        logging.error(f"Error during Prompt 5 execution: {e}", exc_info=True)

    # --- Construct Final Structured Output ---
    final_result = {
        "JOB_TITLE": job_title,
        "COMPANY": company,
        "URL": url, # Use the LLM-extracted URL as it might be cleaner than input URL
        "LOCATION": location,
        "JOB_DESCRIPTION": job_description_extracted, # Use the LLM-extracted description
        "CHALLENGE_AND_ROOT_CAUSE": challenge_and_root_cause,
        "COVER_LETTER_HOOK": cover_letter_hook,
        "COVER_LETTER": cover_letter,
        "TELL_ME_ABOUT_YOURSELF": tell_me_about_yourself,
        "MAIN_CHANGES_TO_MY_CV": main_changes_to_my_cv,
        "QUESTIONS_TO_ASK": questions_to_ask
    }

    logging.info("All prompts executed. Final structured output prepared.")
    return final_result


# --- Main Execution Block (for local testing of url_analyzer.py in isolation) ---
if __name__ == "__main__":
    test_url = input("Enter the URL of the job posting to analyze: ")
    if not test_url.startswith('http'):
        print("Please enter a full URL starting with http:// or https://")
        exit()

    cv_file_path = input("Enter the full path to your CV file (e.g., C:\\Users\\You\\Documents\\MyCV.docx) (leave blank to skip CV analysis): ")

    cv_content = None
    if cv_file_path:
        cv_content = parse_cv_document(cv_file_path)
        if cv_content is None:
            print(f"Failed to parse CV from '{cv_file_path}'. Analysis will proceed without CV content.")

    print(f"\n--- Starting job posting analysis for: {test_url} ---")

    scraped_content = scrape_url_content(test_url) # Removed await for sync main block

    if scraped_content:
        print("\n--- RAW SCRAPED CONTENT (for review): ---")
        print(scraped_content)
        print("\n--- END OF RAW SCRAPED CONTENT ---")

        print("\n--- Content Scraped successfully. Analyzing with Gemini... ---")
        analysis_result = analyze_job_posting_with_gemini(scraped_content, test_url, cv_content) # Removed await for sync main block

        print("\n--- Final Structured Analysis Result: ---")
        print(json.dumps(analysis_result, indent=4, ensure_ascii=False))
    else:
        print("\n--- Failed to scrape job posting content. Cannot proceed with Gemini analysis. ---")
