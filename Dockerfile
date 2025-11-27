# We use the official Playwright image. 
# It comes with Python AND the Chrome browser pre-installed.
FROM mcr.microsoft.com/playwright/python:v1.55.0-jammy


# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file we just created
COPY requirements.txt .

# Install dependencies (FastAPI, OpenAI, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code (main.py, agent.py)
COPY . .

# Run the server
# We bind to 0.0.0.0 so the outside world can reach it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#fix