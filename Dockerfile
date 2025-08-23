FROM python:3.11-slim

WORKDIR /Borderless_Agent

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python files
COPY *.py ./

# Copy data directory
COPY data/ ./data/

# Copy environment file
COPY .env .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]