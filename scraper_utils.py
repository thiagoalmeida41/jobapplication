import logging
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import Optional

# --- Playwright Imports ---
from playwright.async_api import async_playwright, Playwright, TimeoutError as PlaywrightTimeoutError

# --- Logging Configuration (can be inherited from main, but good to have) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Web Scraping Function (FROM url_analyzer.py, now in scraper_utils.py) ---
async def scrape_url_content(url: str) -> Optional[str]:
    """
    Fetches a URL and scrapes visible text content from it, using Playwright for dynamic content.
    Optimized for resource-constrained environments and made asynchronous.
    """
    logging.info(f"Attempting to scrape URL: {url} with Playwright (optimized & async).")

    parsed_url = urlparse(url)
    if not all([parsed_url.scheme, parsed_url.netloc]):
        logging.error(f"Invalid URL format: {url}. Please provide a complete URL including scheme (e.g., 'https://').")
        return None

    page_content = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox', # Required for container environments
                    '--disable-gpu',
                    '--single-process',
                    '--disable-dev-shm-usage',
                    '--disable-setuid-sandbox',
                    '--no-zygote',
                    '--disable-accelerated-2d-canvas'
                ]
            )
            page = await browser.new_page()
            
            await page.set_viewport_size({"width": 800, "height": 600}) 

            def handle_route(route):
                if route.request.resource_type in ["image", "media", "font", "stylesheet"]:
                    route.abort()
                else:
                    route.continue_() 

            await page.route("**/*", handle_route)


            await page.goto(url, wait_until='domcontentloaded', timeout=45000) 
            
            job_desc_selectors = [
                'div[data-automation-id="jobPostingDescription"]', # Workday
                'div.jobDetails', # Workday fallback
                'div[data-ui="job-description"]', # Ashby
                '.ashby-job-posting__content', # Ashby fallback
                'div#content', # Greenhouse
                '.job-description', # General
                'body' # Absolute fallback
            ]
            
            try:
                await page.wait_for_selector(
                    "|".join(job_desc_selectors), 
                    timeout=15000 
                )
                logging.info("Playwright: Job description selector found, content likely rendered.")
            except PlaywrightTimeoutError:
                logging.warning("Playwright: Job description selector not found within timeout. Proceeding with current content.")


            page_content = await page.content() 
            await browser.close()
            logging.info(f"Successfully fetched rendered content from {url} with Playwright.")

    except PlaywrightTimeoutError as e:
        logging.error(f"Playwright navigation/wait timed out for {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected Playwright error occurred during scraping {url}: {e}", exc_info=True)
        return None

    if not page_content:
        logging.error(f"Playwright returned empty content for {url}.")
        return None

    soup = BeautifulSoup(page_content, 'html.parser')

    for unwanted_tag_selector in [
        'script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'button',
        'img', 'svg', 'iframe', 'noscript', 'meta', 'link', 'title', 'head',
        '.header', '.footer', '.navbar', '.sidebar', '.ad', '.ads', '.cookie-banner',
        '.modal', '.overlay', '.share-buttons', '.social-media', '.pagination',
        '.breadcrumb', '.skip-link', '#skip-link', '#footer', '#header', '#navbar',
        '.top-card-layout__card',
        '.sub-nav', '.global-footer', '.sign-in-banner',
        'svg', 'path', 'circle', 'img',
        'div[aria-hidden="true"]',
        '[role="banner"]', '[role="navigation"]', '[role="contentinfo"]',
        '.skip-to-content', '.visually-hidden',
        '.hidden', '.sr-only'
    ]:
        try:
            if unwanted_tag_selector.startswith('.') or unwanted_tag_selector.startswith('#') or '[' in unwanted_tag_selector:
                for element in soup.select(unwanted_tag_selector):
                    element.decompose()
            else:
                for element in soup.find_all(unwanted_tag_selector):
                    element.decompose()
        except Exception as e:
            logging.warning(f"Error decomposing selector {unwanted_tag_selector}: {e}")


    content_element = None
    for selector in job_desc_selectors:
        content_element = soup.select_one(selector)
        if content_element:
            logging.info(f"Final BeautifulSoup: Found job description via selector: {selector}")
            break

    extracted_text = ""
    if content_element:
        extracted_text = content_element.get_text(separator='\n', strip=True)
    else:
        logging.warning("Final BeautifulSoup: No specific job description element found after Playwright render, attempting full body text.")
        body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        extracted_text = body_text

    extracted_text = re.sub(r'[\n\r]+', '\n', extracted_text).strip()
    extracted_text = re.sub(r'[ \t]+', ' ', extracted_text).strip()

    max_final_char_limit = 45000 
    if len(extracted_text) > max_final_char_limit:
        logging.warning(f"Final extracted content length ({len(extracted_text)}) exceeds {max_final_char_limit} characters. Truncating for Google Sheet/LLM.")
        extracted_text = extracted_text[:(max_final_char_limit - 100)] + "\n\n[CONTENT TRUNCATED DUE TO LENGTH LIMITS]..."

    if not extracted_text:
        logging.error(f"No meaningful text extracted from {url} after Playwright and BeautifulSoup processing.")
        return None

    logging.info(f"Successfully scraped and processed content from {url}. Length: {len(extracted_text)} characters.")
    return extracted_text
