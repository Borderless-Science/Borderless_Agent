# server/vectorstore/store.py
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import VECTOR_DB_PATH, EMBEDDING_MODEL_NAME
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Initialize embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

def get_embedding_model():
    """
    Lazily create and return a GoogleGenerativeAIEmbeddings instance.
    This prevents event-loop errors when running in threads without an asyncio loop.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)

embedding_model = get_embedding_model()

def save_vector_store(docs, path=VECTOR_DB_PATH):
    """Creates and saves FAISS vector store from documents."""
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(path)

def load_vector_store(path=VECTOR_DB_PATH):
    """Loads the FAISS vector store."""
    return FAISS.load_local(
        path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
