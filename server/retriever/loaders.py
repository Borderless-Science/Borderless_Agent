# server/retriever/loaders.py
import requests
from typing import List
from langchain_community.document_loaders import ArxivLoader
from langchain.docstore.document import Document

class PubmedLoader:
    def __init__(self, query: str, max_results: int = 10):
        self.query = query
        self.max_results = max_results

    def load(self) -> List[Document]:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        search_url = f"{base_url}esearch.fcgi?db=pubmed&term={self.query}&retmax={self.max_results}&retmode=json"
        search_resp = requests.get(search_url)
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not id_list:
            return []

        ids = ",".join(id_list)
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=text&rettype=abstract"
        fetch_resp = requests.get(fetch_url)

        raw_text = fetch_resp.text.strip()
        entries = [entry.strip() for entry in raw_text.split("\n\n") if len(entry.strip()) > 250]

        documents = []
        for entry in entries:
            lines = entry.splitlines()
            title_line = lines[0].strip() if lines else "Untitled PubMed Entry"
            title = title_line if title_line else "Untitled PubMed Entry"
            documents.append(Document(page_content=entry, metadata={"title": title}))
        return documents


class ArxivResearchLoader:
    def __init__(self, query: str, max_results: int = 5):
        self.query = query
        self.max_results = max_results

    def load(self) -> List[Document]:
        loader = ArxivLoader(query=self.query, max_results=self.max_results)
        return loader.load()
