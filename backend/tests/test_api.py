import pytest
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from backend.src.rag_service import load_and_index_documents

@pytest.fixture
def clean_collection():
    """Clean up ChromaDB collection before/after tests."""
    client_settings = {
        "host": os.getenv("CHROMA_HOST", "chromadb"),
        "port": os.getenv("CHROMA_PORT", "8000")
    }
    vectorstore = Chroma(
        collection_name="rag_collection",
        embedding_function=OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        ),
        client_settings=client_settings
    )
    try:
        vectorstore.delete_collection()
    except:
        pass
    yield
    try:
        vectorstore.delete_collection()
    except:
        pass

def test_load_and_index_documents_openai(clean_collection):
    """Test document loading, chunking, and indexing with OpenAI embeddings."""
    vectorstore = load_and_index_documents(use_openai=True)
    
    assert isinstance(vectorstore, Chroma), "Vectorstore should be a Chroma instance"
    
    # Verify documents indexed
    collection = vectorstore._collection
    assert collection.count() > 0, "Collection should contain documents"
    
    # Test retrieval
    results = vectorstore.similarity_search("security vulnerabilities", k=3)
    assert len(results) > 0, "Should retrieve at least one document"
    assert any("security" in doc.page_content.lower() for doc in results), "Retrieved documents should be relevant"

def test_load_and_index_documents_google(clean_collection, monkeypatch):
    """Test document loading, chunking, and indexing with Google GenAI embeddings."""
    # Mock GOOGLE_API_KEY if not set
    if not os.getenv("GOOGLE_API_KEY"):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    
    try:
        vectorstore = load_and_index_documents(use_openai=False)
        assert isinstance(vectorstore, Chroma), "Vectorstore should be a Chroma instance"
        collection = vectorstore._collection
        assert collection.count() > 0, "Collection should contain documents"
        results = vectorstore.similarity_search("security vulnerabilities", k=3)
        assert len(results) > 0, "Should retrieve at least one document"
        assert any("security" in doc.page_content.lower() for doc in results), "Retrieved documents should be relevant"
    except Exception as e:
        if "test-key" in str(e):
            pytest.skip("Skipping Google test due to invalid API key")
        else:
            raise e

def test_empty_directory(clean_collection):
    """Test behavior with an empty document directory."""
    # Move PDFs to a temporary directory
    os.makedirs("/tmp/empty_pdfs", exist_ok=True)
    for pdf in os.listdir(os.getenv("DATA_DIR")):
        shutil.move(
            f"{os.getenv('DATA_DIR')}/{pdf}",
            f"/tmp/empty_pdfs/{pdf}"
        )

    # Run with empty directory
    vectorstore = load_and_index_documents(use_openai=True)
    collection = vectorstore._collection
    assert collection.count() == 0, "Collection should be empty with no documents"

    # Restore PDFs
    for pdf in os.listdir("/tmp/empty_pdfs"):
        shutil.move(
            f"/tmp/empty_pdfs/{pdf}",
            f"{os.getenv('DATA_DIR')}/{pdf}"
        )

def test_invalid_openai_api_key(clean_collection, monkeypatch):
    """Test behavior with an invalid OpenAI API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-key")
    with pytest.raises(Exception, match=".*authentication.*|.*API key.*"):
        load_and_index_documents(use_openai=True)