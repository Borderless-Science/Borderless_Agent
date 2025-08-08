# core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
VECTOR_DB_PATH = "vector_db"

# Embedding model name (optional config)
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
