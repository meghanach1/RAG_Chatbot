# ingest.py
import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_REPO = os.getenv("CHROMA_REPO", "./chroma_db")
use_openai = True

# -----------------------------
# 1️⃣ Load documents
# -----------------------------
data_folder = "data/docs"
docs = []

for file in os.listdir(data_folder):
    filepath = os.path.join(data_folder, file)
    if file.endswith(".txt"):
        loader = TextLoader(filepath)
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        continue
    docs.extend(loader.load())

print(f"Total documents loaded: {len(docs)}")

# -----------------------------
# 2️⃣ Split into chunks
# -----------------------------
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
print(f"Total chunks: {len(split_docs)}")

# -----------------------------
# 3️⃣ Create embeddings
# -----------------------------
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -----------------------------
# 4️⃣ Setup Chroma
# -----------------------------
client = PersistentClient(path=CHROMA_REPO)
collection_name = "kb_collection"

# Delete existing collection to avoid dimension mismatch
try:
    client.delete_collection(collection_name)
    print("Deleted existing collection to avoid dimension mismatch")
except:
    pass

collection = client.create_collection(collection_name)

# -----------------------------
# 5️⃣ Add documents to Chroma
# -----------------------------
batch_size = 20
for i in range(0, len(split_docs), batch_size):
    batch_docs = split_docs[i:i+batch_size]
    texts = [doc.page_content for doc in batch_docs]
    
    embs = embeddings_model.embed_documents(texts)

    collection.add(
        documents=texts,
        metadatas=[{"source": getattr(doc, "source", "unknown")} for doc in batch_docs],
        embeddings=embs,
        ids=[str(i + idx) for idx, _ in enumerate(batch_docs)]
    )

print("✅ All documents ingested successfully!")
