# prompt construction
"""
Central place to manage prompts for RAG
"""

RAG_PROMPT = """
You are a helpful, concise assistant. Use ONLY the information in the provided context to answer the user's question.
If the answer is not contained in the context, say "I don't know" and suggest where to look.

Context passages:
{context}

User question:
{question}

Answer:
"""
