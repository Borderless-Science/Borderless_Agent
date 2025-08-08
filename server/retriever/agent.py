# server/retriever/agent.py
from crewai import Agent
from server.retriever.loaders import ArxivResearchLoader, PubmedLoader
from server.vectorstore.store import save_vector_store
from core.utils import get_text_splitter

class ResearchRetrieverAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Research Retriever Agent",
            role="Fetches and indexes academic documents from arXiv or PubMed.",
            goal="Vectorize research content for downstream querying.",
            backstory="A scholarly agent that ensures topically relevant literature is available to aid deeper reasoning."
        )

    def retrieve_documents(self, source: str, topic: str) -> int:
        if source.lower() == "arxiv":
            loader = ArxivResearchLoader(query=topic, max_results=5)
        elif source.lower() == "pubmed":
            loader = PubmedLoader(query=topic, max_results=5)
        else:
            raise ValueError(f"Unknown source: {source}")

        docs = loader.load()
        splitter = get_text_splitter()
        split_docs = splitter.split_documents(docs)

        save_vector_store(split_docs)
        return len(split_docs)
