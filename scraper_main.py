import logging
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional

# Import the scraping function from a new utility file
from scraper_utils import scrape_url_content

# Configure logging for the API application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Initialization for Scraper ---
app = FastAPI(
    title="Job Posting Scraper API",
    description="API to scrape job posting content from a given URL using Playwright.",
    version="1.0.0"
)

# --- API Endpoint: Scrape Job Posting ---
@app.post("/scrape_job_content/")
async def scrape_job_content_endpoint(
    url: str = Form(..., description="The URL of the job posting to scrape.")
):
    """
    Scrapes the full text content from a given job posting URL.
    """
    logging.info(f"Received request to scrape URL: {url}")

    try:
        # AWAIT the asynchronous scrape_url_content function
        scraped_content = await scrape_url_content(url)

        if not scraped_content:
            logging.error(f"Failed to retrieve job posting content from URL: {url}")
            raise HTTPException(status_code=500, detail="Failed to retrieve job posting content from the provided URL.")
        
        logging.info(f"Successfully scraped content from {url}. Returning to client.")
        return JSONResponse(content={"job_description_raw": scraped_content})
    except Exception as e:
        logging.error(f"Error during scraping process for {url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during scraping: {str(e)}")

# Optional: A simple root endpoint to check if the API is running
@app.get("/")
async def read_root():
    return {"message": "Job Posting Scraper API is running!"}

