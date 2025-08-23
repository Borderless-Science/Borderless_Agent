FROM python:3.11-slim

WORKDIR /Borderless_Agent

# Install system dependencies
#RUN apt-get update && apt-get install -y \
#    gcc \
#    && rm -rf /var/lib/apt/lists/

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


# Copy application code
#COPY Borderless_Agent/app/ .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.fast:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]