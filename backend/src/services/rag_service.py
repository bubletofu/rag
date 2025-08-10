# backend/src/services/rag_service.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # You can swap with SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "/Users/phuong/Desktop/rag/database/pdfs")
PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "/app/.index_storage")

def load_and_index_documents(use_openai=True):

    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = pdf_loader.load()
    print(f"[INFO] Loaded {len(docs)} documents")

    for doc in docs:
        raw_path = doc.metadata.get("source", doc.metadata.get("file_path", "unknown"))
        doc.metadata["source"] = os.path.basename(raw_path)

    embed_model = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    #sematic chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks created: {len(chunks)}")


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=PERSIST_DIR,
        client_settings=Settings(persist_directory=PERSIST_DIR)
    )

    vectorstore.persist()
    print(f"vector stored {PERSIST_DIR}")
    return vectorstore



if __name__ == "__main__":
    import time

    print("\n[TEST] Running load_and_index_documents()...\n")
    start = time.time()

    try:
        vectorstore = load_and_index_documents()
        # for doc in docs:
        #     print(f"[DOC] {doc.metadata['source']} – {len(doc.page_content)} characters")
        print(f"[TEST SUCCESS] Indexed {vectorstore._collection.count()} vectors.")

    except Exception as e:
        print(f"[TEST FAILED] {e}")

    end = time.time()
    print(f"[INFO] Elapsed time: {end - start:.2f} seconds\n")