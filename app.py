import streamlit as st
from src.ingest import ingest_documents
from src.retriever import load_retriever
from src.chain import build_chain
import os

st.set_page_config(page_title="Indic Knowledge RAG", page_icon="🕉️")
st.title("🕉️ Indic Knowledge Chatbot")
st.caption("Ask anything from the Bhagavad Gita")

# Ingest once
if not os.path.exists("chroma_db"):
    with st.spinner("Loading and indexing the Gita... this takes a minute on first run."):
        ingest_documents("data/The Bhagavad Gita.pdf")

# Load retriever and chain
retriever = load_retriever()
chain = build_chain(retriever)

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the Gita..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})