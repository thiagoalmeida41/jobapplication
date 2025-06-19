# Start with a base image that is specifically designed for Playwright.
# This image includes Python, Playwright, and all necessary browser binaries.
# Using a specific version (e.g., v1.44.0-jammy) for stability.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Set the working directory in the container
WORKDIR /app

# Copy your Python dependencies file into the container
COPY requirements.txt .

# Install your Python application dependencies.
# Playwright and its browsers are already present in the base image.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run your FastAPI application with Uvicorn
# Ensure main:app matches your main.py file and FastAPI app instance name
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]