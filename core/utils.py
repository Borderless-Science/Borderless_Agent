# core/utils.py
from langchain.text_splitter import CharacterTextSplitter

def get_text_splitter(chunk_size=1000, chunk_overlap=100):
    """Returns a configured text splitter."""
    return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
