# RAG Chatbot — Python + LLM + Chromadb (Beginner-friendly)

A simple Retrieval-Augmented Generation (RAG) chatbot:
- Ingests documents (PDF / TXT) from `data/docs/`
- Creates embeddings and stores them in a local Chromadb vector store
- Runs a Streamlit UI that retrieves relevant passages and asks an LLM to generate a grounded answer

## Features
- Document ingestion (PDF & TXT)
- Chunking & embeddings
- Chromadb local persistence
- Simple RAG prompt template & citation
- Streamlit UI

## Prerequisites
- Python 3.10+
- OpenAI API key (or change to another provider in code)
- (Optional) Docker

## Quick start

1. Clone or create project folder:
```bash
mkdir rag-chatbot
cd rag-chatbot
# create files from this repo
Create and activate a venv:

bash
Copy code
python -m venv .venv
source .venv/bin/activate    # mac/linux
# .venv\Scripts\activate     # windows
Add the files from this repo into the folder (see file list).

Install dependencies:

bash
Copy code
pip install -r requirements.txt
Create .env file from .env.example and add your OpenAI API key:

bash
Copy code
cp .env.example .env
# then edit .env and add OPENAI_API_KEY
Put some documents in data/docs/ (PDF or .txt). Example: example.pdf, notes.txt.

Run ingestion (build vector DB):

bash
Copy code
python ingest.py
Launch UI:

bash
Copy code
streamlit run app_streamlit.py
# open the URL printed by Streamlit (usually http://localhost:8501)
Files
ingest.py - ingest docs and build Chromadb collection

retriever.py - retrieval + generation

app_streamlit.py - Streamlit UI

utils.py - helpers for chunking & text extraction

