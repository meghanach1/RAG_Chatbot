# retriever.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_REPO = os.getenv("CHROMA_REPO", "./chroma_db")

client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = PersistentClient(path=CHROMA_REPO)

try:
    collection = chroma_client.get_collection("kb_collection")
except:
    collection = chroma_client.create_collection("kb_collection")


def retrieve(query, k=4):
    # Get embedding for query
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    q_emb = resp.data[0].embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    return docs, metas


PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following retrieved document passages to answer the user's question. 
If the answer cannot be found in the passages, say you don't know and suggest how to find out.

==========
{context}
==========

User question: {question}

Answer concisely and cite the source filenames where relevant.
"""


def answer_question(question):
    docs, metas = retrieve(question, k=4)
    context = "\n\n---\n\n".join(
        [f"Source: {m.get('source')}\n\n{d}" for d, m in zip(docs, metas)]
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Call Chat LLM
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1
    )
    answer = completion.choices[0].message.content
    return answer, docs, metas
