# server/query/agent.py
from crewai import Agent
from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI
from server.vectorstore.store import load_vector_store
from core.config import GOOGLE_API_KEY

# Initialize Google Generative AI client
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", max_tokens=500, api_key=GOOGLE_API_KEY)

class ResearchQueryAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Research Query Agent",
            role="Answers academic research questions based on vectorized content.",
            goal="Deliver fact-based answers from arXiv or PubMed sources.",
            backstory="An academic expert that distills key insights from scientific articles."
        )

    def detect_language(self, query: str) -> str:
        try:
            return detect(query)
        except:
            return "en"

    def answer_query(self, query: str) -> str:
        user_language = self.detect_language(query)
        vector_store = load_vector_store()
        relevant_docs = vector_store.similarity_search(query, k=3)

        combined_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""
        You are an academic assistant AI. Use the following research documents to answer the user's question.

        --------------------
        {combined_text}
        --------------------

        User Query: {query}
        Respond in English only.
        """

        response = llm.invoke(
            input=[{"role": "user", "content": prompt}]
        )

        return response.content.strip()
