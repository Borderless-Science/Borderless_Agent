# client/ui/interface.py
import streamlit as st
from server.retriever.agent import ResearchRetrieverAgent
from server.query.agent import ResearchQueryAgent
from server.retriever.loaders import ArxivResearchLoader, PubmedLoader

# UI Config
st.set_page_config(page_title="Borderless AI Agent", layout="wide")
st.title("🔬 Research-Based AI Assistant")
st.write("Search academic literature and get AI-powered answers from arXiv or PubMed using structured components.")

# Input Section
source = st.selectbox("📚 Select Research Source:", ["arXiv", "PubMed"])
topic = st.text_input("📌 Enter Topic to Fetch Documents:")
query = st.text_area(" Ask Your Research Question:")

# Fetch Button
if st.button("📥 Fetch & Index Documents") and topic:
    retriever = ResearchRetrieverAgent()
    count = retriever.retrieve_documents(source, topic)
    st.success(f"✅ Indexed {count} documents from {source}.")

    # Show Document Titles
    with st.expander("📄 View Fetched Documents"):
        if source == "arXiv":
            loader = ArxivResearchLoader(topic)
        else:
            loader = PubmedLoader(topic)
        docs = loader.load()
        for i, doc in enumerate(docs):
            title = doc.metadata.get("title", f"Document {i+1}")
            st.markdown(f"- {title}")

# Answer Button
if st.button("🤖 Get Answer") and query:
    responder = ResearchQueryAgent()
    response = responder.answer_query(query)
    st.subheader("💡 AI Response:")
    st.write(response)
