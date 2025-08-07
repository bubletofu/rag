#backend/src/services/rag_service.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings 
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "/Users/phuong/Desktop/rag/database/pdfs")
PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "/app/.index_storage") 

def load_and_index_documents(use_openai=True):
    #Load PDF
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = pdf_loader.load()

    for doc in docs:
        raw_path = doc.metadata.get("source", doc.metadata.get("file_path", "unknown"))
        doc.metadata["source"] = os.path.basename(raw_path)  

    #Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    #Embeddings
    embed_model = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    #Store in vectorstore
    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory=PERSIST_DIR,
        client_settings=Settings(
            persist_directory=PERSIST_DIR
        )
    )

    return vectorstore


