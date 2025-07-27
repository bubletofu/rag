"""
backend/src/services/rag_service.py

RAG service using LlamaIndex + Google GenAI (Gemini)
"""
import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

BASE_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../../.env"))

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Paths
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "../../database/texts"))
PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", os.path.join(BASE_DIR, "../../.index_storage"))

# Verify directories
if not os.path.exists(DATA_DIR):
    raise ValueError(f"DATA_DIR does not exist: {DATA_DIR}")
if not os.path.exists(PERSIST_DIR):
    raise ValueError(f"INDEX_PERSIST_DIR does not exist: {PERSIST_DIR}")
if not os.listdir(DATA_DIR):
    raise ValueError(f"DATA_DIR is empty: {DATA_DIR}")

# Global Settings configuration
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-005",
    api_key=google_api_key
)

Settings.llm = GoogleGenAI(
    model="gemini-1.5-flash",
    api_key=google_api_key,
    temperature=0
)

def build_or_load_index() -> VectorStoreIndex:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    if os.listdir(PERSIST_DIR):
        return load_index_from_storage(storage_ctx)
    docs = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)
    storage_ctx.persist()
    return index

def answer_query(query: str, top_k: int = 3) -> str:
    index = build_or_load_index()
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query)
    return response.response

if __name__ == "__main__":
    print(answer_query("What is the main topic of the documents?", top_k=2))