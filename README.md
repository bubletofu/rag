# RAG Pipeline: Full Implementation Guide

This document outlines a complete Retrieval-Augmented Generation (RAG) pipeline using Python, **LangChain**, FastAPI, PostgreSQL+pgvector or ChromaDB, and Google GenAI (Gemini) for embeddings and LLM generation.

## Tech Stack & Tools
- **Python 3.9+**
- **LangChain** for data ingestion, chunking, retrieval, and query chains
- **Google GenAI Embeddings** (`GoogleGenerativeAIEmbeddings`) for vectorization
- **Google GenAI LLM** (`ChatGoogleGenerativeAI`) for answer generation
- **FastAPI** for building the backend API
- **Uvicorn** as the ASGI server
- **ChromaDB** or **PostgreSQL + pgvector** for vector storage
- **Mermaid** for architecture & sequence diagrams in Markdown
- **Docker** & **docker-compose** for containerization

## System Architecture

```mermaid
flowchart LR
  subgraph Ingestion
    A["Raw Docs (PDF, HTML, Crawled)"] --> B["Text Loader (LangChain Loaders)"]
    B --> C["Chunker (RecursiveCharacterTextSplitter)"]
  end

  subgraph Embedding
    C --> D["GoogleGenerativeAIEmbedding Task"]
    D -->|enqueue| CW1["Celery Worker: Embedding"]
  end

  subgraph Storage
    CW1 --> E["ChromaDB"]
  end

  subgraph Backend
    E --> F["Retriever (VectorStoreRetriever)"]
    F --> G["LLM Generation Task (RetrievalQA Chain)"]
    G -->|enqueue| CW2["Celery Worker: LLM Generator"]
    CW2 --> H["FastAPI `/query` Endpoint"]
  end

  subgraph MessageBroker
    MQ1["Redis or RabbitMQ"]
    MQ1 --> CW1
    MQ1 --> CW2
  end

  subgraph Frontend
    J["React/Vue UI"] --> H
    H --> J
  end
```

## API Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend
    participant API as FastAPI Backend
    participant RAG as RAG Service
    box RAG
        participant RET as Retriever
        participant GEN as Generator
    end
    participant VS as Vector Store
    participant LLM as Google GenAI

    %% --- Document Loading Phase ---
    U->>FE: Upload document
    FE->>API: POST /load-document { file }
    API->>RAG: load_document(file)
    RAG->>RET: chunk_and_embed(file)
    RET->>VS: store_embeddings(chunks, embeddings)
    note right of VS: Vector store is initialized<br/>on first load

    %% --- Document Query Phase ---
    U->>FE: Enter question & submit
    FE->>FE: validate_input() & show_loading()
    FE->>API: POST /query { query, top_k }

    API->>RAG: answer_query(query, top_k)
    RAG->>RET: similarity_search(query, top_k)
    RET->>VS: query_embeddings(query, top_k)
    VS-->>RET: return top_k chunks with scores

    RAG->>GEN: generate_answer(chunks, query)
    GEN-->>LLM: call Google GenAI API (async)
    LLM-->>GEN: return generated_text

    GEN-->>RAG: format_response(query, generated_text, chunks, scores)
    RAG-->>API: return JSON response
    note right of API: Includes answer, query,<br/>retrieved chunks and scores

    API-->>FE: Return JSON Response
    FE->>FE: display_answer(response)

    FE-->>U: Show answer, metadata

    %% --- Error Handling Path ---
    alt Any failure occurs
        RAG-->>API: error { message: "timeout" }
        API-->>FE: HTTP 500 { error: "Internal Server Error" }
        FE->>FE: show_error_to_user()
    end
```

## Implementation Outline

1. **Setup & Dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install fastapi uvicorn langchain langchain-community langchain-google-genai chromadb psycopg2-binary python-dotenv
    ```

2. **Data Ingestion & Chunking**
    ```python
    from langchain_community.document_loaders import DirectoryLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = DirectoryLoader("database/texts", glob="**/*.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    ```

3. **Embedding & Indexing**
    ```python
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import Chroma
    import os

    embed_model = GoogleGenerativeAIEmbeddings(model="text-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embed_model, persist_directory="./.index_storage")
    vectorstore.persist()
    ```

4. **Backend API**
    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA

    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        top_k: int = 3

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
    vectorstore = Chroma(persist_directory="./.index_storage", embedding_function=GoogleGenerativeAIEmbeddings(model="text-embedding-001"))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    @app.post("/query")
    def query_endpoint(req: QueryRequest):
        result = qa_chain.run(req.query)
        return {"answer": result}
    ```

5. **Docker Compose**
    ```yaml
    version: "3.8"
    services:
      backend:
        build: ./backend
        ports:
          - "8000:8000"
        volumes:
          - ./database/texts:/app/database/texts
          - ./backend/.index_storage:/app/.index_storage

      vectorstore:
        image: postgres:15
        environment:
          POSTGRES_DB: rag
          POSTGRES_USER: rag
          POSTGRES_PASSWORD: rag
        volumes:
          - ./postgres-data:/var/lib/postgresql/data
    ```