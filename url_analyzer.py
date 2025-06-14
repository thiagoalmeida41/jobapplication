import requests
from bs4 import BeautifulSoup
import logging
import re
from urllib.parse import urlparse
import google.generativeai as genai
import os
import json
from docx import Document
from docx.opc.exceptions import OpcError
from typing import Optional, List, Dict # Added for type hinting clarity

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini API Configuration ---
# Use GOOGLE_API_KEY as per your existing code.
# Change: Replaced exit() with raising an exception so FastAPI can catch it.
# This prevents the Render service from crashing on startup if the key is missing.
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
    logging.info("Gemini API key loaded from environment variable.")
except ValueError as e:
    logging.error(f"Configuration error: {e}")
    # In a deployed FastAPI app, raising a RuntimeError here allows the server to start
    # but requests using the model will fail, which is better than a hard exit.
    # The analyze_job_posting_with_gemini function will also handle this if genai.configure fails.
    raise RuntimeError(f"Gemini API configuration failed: {e}. Please ensure GOOGLE_API_KEY is set.")
except Exception as e:
    logging.error(f"An unexpected error occurred during Gemini API configuration: {e}", exc_info=True)
    raise RuntimeError(f"Gemini API configuration failed unexpectedly: {e}")


# Initialize the Gemini model globally
# 'gemini-1.5-flash' is a good balance of speed and capability for many tasks.
# You could use 'gemini-1.5-pro' for more complex reasoning, but it might be slower/pricier.
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Test a small call to ensure model is accessible (optional, can be slow on startup)
    # response = model.generate_content("ping", safety_settings={'HARASSMENT': 'BLOCK_NONE'})
    # logging.info(f"Gemini model initialized successfully: {response.text.strip()}")
except Exception as e:
    logging.error(f"Failed to initialize Gemini model 'gemini-1.5-flash': {e}", exc_info=True)
    # Re-raise to ensure the app doesn't start if the model can't be loaded
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")


# --- Web Scraping Function ---
def scrape_url_content(url: str) -> Optional[str]: # Changed return type hint to Optional[str]
    """
    Fetches a URL and scrapes visible text content from it.
    This version aims to capture as much content as possible while
    removing common non-content tags and cleaning whitespace.
    """
    logging.info(f"Attempting to scrape URL: {url}")

    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        logging.error(f"Invalid URL format: {url}. Please provide a complete URL including scheme (e.g., 'https://').")
        return None

    try:
        # Set a User-Agent header to mimic a browser and avoid some blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Make the HTTP GET request with a timeout
        response = requests.get(url, headers=headers, timeout=20) # Increased timeout slightly
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        logging.info(f"Successfully fetched content from {url} (Status: {response.status_code})")

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find specific common content containers first for better accuracy
        # Change: Added more specific selectors common on job boards.
        # Prioritize these before aggressive general cleanup.
        content_element = None
        for selector in [
            'div#content',                 # Common for Greenhouse.io
            'div.job-description',         # Common for many sites
            'section.job-details',
            'div[data-qa="job-description"]', # Common for some job boards
            'div.description',
            'div.full-description',
            'div[itemprop="description"]', # Microdata standard
            'article.job-content'
        ]:
            content_element = soup.select_one(selector)
            if content_element:
                logging.info(f"Found content via selector: {selector}")
                break

        # If a specific content element is found, work within it.
        # Otherwise, perform more general filtering on the whole body.
        if content_element:
            target_soup = content_element
            logging.info("Scraping within specific content element.")
        else:
            target_soup = soup.body if soup.body else soup
            logging.warning("No specific job description element found, scraping entire body/document.")


        # Remove elements that typically don't contain main content from the target_soup
        # or are navigation/boilerplate. This list is expanded for more aggressive filtering.
        for unwanted_tag_selector in [
            'script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button',
            'img', 'svg', 'iframe', 'noscript', 'meta', 'link', 'title', 'head',
            # Common classes/ids for small, repetitive, or non-content elements
            '.header', '.footer', '.navbar', '.sidebar', '.ad', '.ads', '.cookie-banner',
            '.modal', '.overlay', '.share-buttons', '.social-media', '.pagination',
            '.breadcrumb', '.skip-link', '#skip-link', '#footer', '#header', '#navbar',
            '.top-card-layout__card', # LinkedIn specific top card
            '.sub-nav', '.global-footer', '.sign-in-banner'
        ]:
            if unwanted_tag_selector.startswith('.') or unwanted_tag_selector.startswith('#'):
                for element in target_soup.select(unwanted_tag_selector):
                    element.decompose()
            else: # Assume it's a tag name
                for element in target_soup.find_all(unwanted_tag_selector):
                    element.decompose()

        # Get all remaining visible text, using '\n' as separator to preserve lines/paragraphs
        extracted_text = target_soup.get_text(separator='\n', strip=True)

        # Change: More careful newline consolidation. Preserve single newlines for paragraphs.
        extracted_text = re.sub(r'[\n\r]+', '\n', extracted_text).strip() # Consolidate multiple newlines into single
        extracted_text = re.sub(r'[ \t]+', ' ', extracted_text).strip() # Consolidate multiple spaces/tabs

        # Truncate if the text is extremely long to prevent issues with LLM token limits.
        # Adjusted limit to 15000 characters (approx. 3750 tokens) to allow more content.
        if len(extracted_text) > 15000:
            logging.warning(f"Scraped content length ({len(extracted_text)}) exceeds 15000 characters. Truncating.")
            extracted_text = extracted_text[:14900] + "\n\n[TEXT TRUNCATED DUE TO EXTREME LENGTH]..."


        logging.info(f"Successfully scraped content from {url}. Length: {len(extracted_text)} characters (final extract).")
        return extracted_text

    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout occurred while fetching {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request error for {url}: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during scraping {url}: {e}", exc_info=True)
        return None

# --- Function: Parse CV Document ---
def parse_cv_document(file_path: str) -> Optional[str]:
    """
    Parses a .docx file and extracts its text content.
    Note: This function specifically handles .docx files.
    Older .doc files require different libraries or external tools.
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

# --- Gemini Prompting Functions ---

def analyze_job_posting_with_gemini(content: str, url: str, cv_content: Optional[str]) -> Dict[str, any]:
    """
    Orchestrates a series of Gemini API calls to analyze a job posting
    and generate application-related content, now using CV content.

    Args:
        content (str): The scraped text content from the job posting URL.
        url (str): The original URL of the job posting.
        cv_content (str | None): The text content of the user's CV, or None if not provided.

    Returns:
        dict: A dictionary containing the structured job analysis and
              generated application content.
    """
    # Changed: Ensure a default empty dict for final_result in case of early return/failure
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


    # Define a common generation configuration for JSON output
    # Change: Added response_schema for more robust structured output
    json_generation_config = {
        "response_mime_type": "application/json",
        "response_schema": { # Example schema, refine as needed per prompt output
            "type": "OBJECT",
            "properties": {
                "JOB_TITLE": {"type": "STRING"},
                "COMPANY": {"type": "STRING"},
                "LOCATION": {"type": "STRING"},
                "JOB_DESCRIPTION": {"type": "STRING"},
                # Add other properties as they appear in the JSON output
                # This ensures the model tries to conform
            }
        }
    }


    # --- Prompt 1: Extract Core Job Information (Revised for Broader JOB_DESCRIPTION) ---
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
    job_title, company, location, job_description_extracted = "N/A", "N/A", "N/A", "N/A" # Renamed to avoid conflict
    try:
        response1 = model.generate_content(
            prompt1_extraction,
            generation_config=json_generation_config
        )
        parsed_data = json.loads(response1.text)
        job_title = parsed_data.get("JOB_TITLE", "N/A")
        company = parsed_data.get("COMPANY", "N/A")
        location = parsed_data.get("LOCATION", "N/A")
        job_description_extracted = parsed_data.get("JOB_DESCRIPTION", "N/A") # Renamed
        logging.info(f"Prompt 1 successful. Job Title: {job_title}, Company: {company}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 1: {e}. Raw response: '{response1.text}'")
        # Added more robust markdown wrapper removal
        cleaned_text = response1.text.strip()
        if cleaned_text.startswith("```json") and cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[len("```json"): -len("```")].strip()
            try:
                parsed_data = json.loads(cleaned_text)
                job_title = parsed_data.get("JOB_TITLE", "N/A")
                company = parsed_data.get("COMPANY", "N/A")
                location = parsed_data.get("LOCATION", "N/A")
                job_description_extracted = parsed_data.get("JOB_DESCRIPTION", "N/A") # Renamed
                logging.warning("Recovered from Markdown wrapper in Prompt 1 response.")
            except json.JSONDecodeError:
                logging.error("Failed to recover from Markdown wrapper.")
                pass
        logging.error("Failed to extract core job details, proceeding with N/A values.")
    except Exception as e:
        logging.error(f"Error during Prompt 1 execution: {e}", exc_info=True)
        logging.error("Failed to extract core job details, proceeding with N/A values.")

    # Use the extracted job_description_extracted for subsequent prompts if valid, otherwise use original content
    job_description_for_prompts = job_description_extracted if job_description_extracted != "N/A" and job_description_extracted.strip() else content


    # --- Guard Clause: If job_description is not available, cannot generate further ---
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

    # Prepare CV content for prompts
    cv_context = ""
    if cv_content:
        cv_context = f"\n\nMy CV Content:\n---\n{cv_content}\n---\n"
        logging.info("CV content provided for generation prompts.")
    else:
        logging.warning("No CV content provided or parsed for generation prompts. Results may be less personalized.")


    # --- NEW Prompt: Identify Biggest Challenge and Root Cause ---
    # Changed: Added schema to generation_config for this prompt
    challenge_json_schema = {
        "type": "OBJECT",
        "properties": {
            "CHALLENGE_AND_ROOT_CAUSE": {"type": "STRING"}
        }
    }
    prompt_challenge_and_root_cause = f"""
    Based solely on the job description provided, what is the biggest challenge someone in this "{job_title}" position at "{company}" would face day-to-day?
    After identifying the challenge, give me the root cause of this specific issue.

    Respond strictly in JSON format, conforming to this schema:
    {{
        "CHALLENGE_AND_ROOT_CAUSE": "string"
    }}

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
            generation_config={"response_mime_type": "application/json", "response_schema": challenge_json_schema}
        )
        parsed_data = json.loads(response_challenge.text)
        challenge_and_root_cause = parsed_data.get("CHALLENGE_AND_ROOT_CAUSE", "N/A")
        logging.info("NEW Prompt successful. Challenge and Root Cause identified.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Challenge Prompt: {e}. Raw response: '{response_challenge.text}'")
    except Exception as e:
        logging.error(f"Error during Challenge Prompt execution: {e}", exc_info=True)


    # --- NEW Prompt: Generate Cover Letter Hook ---
    # Changed: Added schema to generation_config for this prompt
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
    Respond strictly in JSON format, conforming to this schema:
    {{
        "COVER_LETTER_HOOK": "string"
    }}

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
            generation_config={"response_mime_type": "application/json", "response_schema": hook_json_schema}
        )
        parsed_data = json.loads(response_hook.text)
        cover_letter_hook = parsed_data.get("COVER_LETTER_HOOK", "N/A")
        logging.info("NEW Prompt successful. Cover Letter Hook generated.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Hook Prompt: {e}. Raw response: '{response_hook.text}'")
    except Exception as e:
        logging.error(f"Error during Hook Prompt execution: {e}", exc_info=True)


    # --- MODIFIED Prompt 2: Generate Full Cover Letter Draft ---
    # Changed: Added schema to generation_config for this prompt
    cover_letter_json_schema = {
        "type": "OBJECT",
        "properties": {
            "COVER_LETTER": {"type": "STRING"}
        }
    }
    prompt2_cover_letter = f"""
    You are writing a cover letter applying for the "{job_title}" role at "{company}".
    Here's what you have so far, keep this word for word:
    {cover_letter_hook}

    Finish writing the cover letter based on your resume and keep the *entire* cover letter (including the hook) within 250 words.
    Focus on connecting your skills and experiences directly to the job description, elaborating on how your background makes you an ideal candidate.
    Do NOT include salutation, closing, or placeholders like "[Your Name]". Focus on the content of the letter.

    {cv_context}
    Respond strictly in JSON format, conforming to this schema:
    {{
        "COVER_LETTER": "string"
    }}

    Job Title: {job_title}
    Company: {company}
    Job Description:
    ---
    {job_description_for_prompts}
    ---
    """
    logging.info("Sending MODIFIED Prompt 2: Generate Full Cover Letter Draft...")
    cover_letter = "N/A"
    try:
        response2 = model.generate_content(
            prompt2_cover_letter,
            generation_config={"response_mime_type": "application/json", "response_schema": cover_letter_json_schema}
        )
        parsed_data = json.loads(response2.text)
        cover_letter = parsed_data.get("COVER_LETTER", "N/A")
        logging.info("Prompt 2 successful. Cover Letter generated.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 2: {e}. Raw response: '{response2.text}'")
    except Exception as e:
        logging.error(f"Error during Prompt 2 execution: {e}", exc_info=True)


    # --- Prompt 3: Generate "Tell Me About Yourself" Response (UPDATED) ---
    # Changed: Added schema to generation_config for this prompt
    tmay_json_schema = {
        "type": "OBJECT",
        "properties": {
            "TELL_ME_ABOUT_YOURSELF": {"type": "STRING"}
        }
    }
    prompt3_tell_me_about_yourself = f"""
    You are a career coach. Based *strictly* on the experiences and qualifications provided in the CV content, and the job description, draft a concise and compelling "Tell me about yourself" response (around 100-150 words).

    Your response must:
    -   **ONLY** use information explicitly mentioned in the provided CV content. Do NOT make up any details or experiences.
    -   Directly connect the candidate's background from the CV to how they can successfully address or contribute to solving the biggest challenge of the role identified as: "{challenge_and_root_cause}".
    -   Structure it like a brief story: present, past, future, *as supported by the CV*.

    If the CV does not contain information directly relevant to the identified challenge, generate a general "Tell Me About Yourself" response based on the CV and job description, clearly stating that direct connection to the challenge isn't evident in the provided CV.

    Respond strictly in JSON format, conforming to this schema:
    {{
        "TELL_ME_ABOUT_YOURSELF": "string"
    }}

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
    Challenge and Root Cause for this role: {challenge_and_root_cause}
    """
    logging.info("Sending Prompt 3: Tell Me About Yourself (UPDATED)...")
    tell_me_about_yourself = "N/A"
    try:
        response3 = model.generate_content(
            prompt3_tell_me_about_yourself,
            generation_config={"response_mime_type": "application/json", "response_schema": tmay_json_schema}
        )
        parsed_data = json.loads(response3.text)
        tell_me_about_yourself = parsed_data.get("TELL_ME_ABOUT_YOURSELF", "N/A")
        logging.info("Prompt 3 successful. 'Tell Me About Yourself' generated.")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from Prompt 3: {e}. Raw response: '{response3.text}'")
    except Exception as e:
        logging.error(f"Error during Prompt 3 execution: {e}", exc_info=True)


    # --- Prompt 4: Suggest Main Changes to CV (UPDATED for Two Columns) ---
    # Changed: Added schema to generation_config for this prompt
    cv_changes_json_schema = {
        "type": "OBJECT",
        "properties": {
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
            }
        }
    }
    prompt4_cv_changes = f"""
    You are a resume expert. Based on the following job description AND the provided CV content, identify 3-5 key areas where an applicant's CV could be optimized to better align with this specific role.

    For each recommendation, you must provide it in two parts:
    1.  The **exact original phrase or sentence** (or a very short bullet point) from the CV that this recommendation pertains to.
    2.  A **proposed updated text** for that original phrase or sentence, tailored to highlight relevance for the job description.

    If the provided CV content is "No CV content provided.", or if no specific phrases can be identified for direct rephrasing, then provide 3-5 general recommendations. In this fallback case, for each recommendation, set "original_cv_text" to "General advice (No CV provided)" and provide a general proposed update in "proposed_update".

    Respond strictly in JSON format, conforming to this schema:
    {{
        "MAIN_CHANGES_TO_MY_CV": [
            {{
                "original_cv_text": "string", // Exact original text from CV or "General advice (No CV provided)"
                "proposed_update": "string"    // Proposed new text
            }},
            // ... up to 5 such objects
        ]
    }}

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
            generation_config={"response_mime_type": "application/json", "response_schema": cv_changes_json_schema}
        )
        parsed_data = json.loads(response4.text)

        # Validate the structure of the returned data for MAIN_CHANGES_TO_MY_CV
        if isinstance(parsed_data.get("MAIN_CHANGES_TO_MY_CV"), list):
            temp_list = []
            for item in parsed_data["MAIN_CHANGES_TO_MY_CV"]:
                # Changed: Check for both keys, as required by the schema
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
        # Attempt to clean up if the model still misbehaves (less likely with response_mime_type)
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
    # Changed: Added schema to generation_config for this prompt
    questions_json_schema = {
        "type": "OBJECT",
        "properties": {
            "QUESTIONS_TO_ASK": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        }
    }
    prompt5_questions_to_ask = f"""
    You are an interview coach. Based on the following job description and company, suggest 3-5 insightful questions a candidate should ask the interviewer.
    Focus on questions that show genuine interest, strategic thinking, or a desire to understand the team/company culture.
    {cv_context}
    Respond strictly in JSON format, conforming to this schema:
    {{
        "QUESTIONS_TO_ASK": ["string", "string", ...]
    }}

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
            generation_config={"response_mime_type": "application/json", "response_schema": questions_json_schema}
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
        "URL": url, # Changed: Ensure URL from function argument is used
        "LOCATION": location,
        "JOB_DESCRIPTION": job_description_extracted, # Changed: Use the LLM-extracted description
        "CHALLENGE_AND_ROOT_CAUSE": challenge_and_root_cause,
        "COVER_LETTER_HOOK": cover_letter_hook,
        "COVER_LETTER": cover_letter,
        "TELL_ME_ABOUT_YOURSELF": tell_me_about_yourself,
        "MAIN_CHANGES_TO_MY_CV": main_changes_to_my_cv,
        "QUESTIONS_TO_ASK": questions_to_ask
    }

    logging.info("All prompts executed. Final structured output prepared.")
    return final_result


# --- Main Execution Block ---
# This block is for direct script execution (e.g., python url_analyzer.py)
# and won't run when imported by main.py (your FastAPI app).
# It's useful for testing url_analyzer.py in isolation.
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

    scraped_content = scrape_url_content(test_url)

    if scraped_content:
        # --- Print scraped content for debugging ---
        print("\n--- RAW SCRAPED CONTENT (for review): ---")
        print(scraped_content)
        print("\n--- END OF RAW SCRAPED CONTENT ---")
        # --- END Debugging ---

        print("\n--- Content Scraped successfully. Analyzing with Gemini... ---")
        analysis_result = analyze_job_posting_with_gemini(scraped_content, test_url, cv_content)

        print("\n--- Final Structured Analysis Result: ---")
        print(json.dumps(analysis_result, indent=4, ensure_ascii=False))
    else:
        print("\n--- Failed to scrape job posting content. Cannot proceed with Gemini analysis. ---")

