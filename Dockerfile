# Start with a base image that is specifically designed for Playwright.
# This image includes Python, Playwright, and all necessary browser binaries.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set the working directory in the container
WORKDIR /app

# Copy your Python dependencies file into the container
COPY requirements.txt .

# Install your Python application dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run your FastAPI application with Uvicorn.
# FIX: Run 'playwright install' as part of the startup command.
# This forces Playwright to set up its browsers in the runtime environment.
CMD ["bash", "-c", "playwright install --with-deps && uvicorn scraper_main:app --host 0.0.0.0 --port 8000"]
