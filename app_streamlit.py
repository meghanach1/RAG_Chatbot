# app_streamlit.py
import streamlit as st
from retriever import answer_question

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot — Ask about your documents")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question about the uploaded documents:", key="input")
if st.button("Ask"):
    if query.strip():
        with st.spinner("Searching and generating answer..."):
            answer, docs, metas = answer_question(query)
        st.session_state.history.append((query, answer))
        st.success("Answer generated.")

for q, a in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown("---")

st.sidebar.header("Controls")
st.sidebar.write("Use ingest.py to index docs in `data/docs/` before asking questions.")
