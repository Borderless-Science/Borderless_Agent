# client/ui/interface.py
import streamlit as st
from server.rag_agent import crew

# UI Config
st.set_page_config(page_title="Borderless AI Agent", layout="wide")
st.title("🔬 Research-Based AI Assistant")
st.write("Search academic literature and get AI-powered answers from arXiv or PubMed using structured components.")

# Input Section
query = st.text_area(" Ask Your Research Question:")



# Answer Button
if st.button("🤖 Get Answer") and query:
    inputs = {
        "query": query
    }
    result = crew.kickoff(inputs=inputs)
    response = result.raw
    st.subheader("💡 AI Response:")
    st.write(response)
