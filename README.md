# RAG Pipeline: Full Implementation Guide

This document outlines a complete Retrieval-Augmented Generation (RAG) pipeline using Python, LlamaIndex, FastAPI, PostgreSQL+pgvector or ChromaDB, and Google GenAI (Gemini) for embeddings and LLM generation.

## Tech Stack & Tools
- **Python 3.9+**
- **LlamaIndex** for data ingestion, chunking, and query engines
- **Google GenAI Embeddings** (`GoogleGenAIEmbedding`) for vectorization
- **Google GenAI LLM** (`GoogleGenAI`) for answer generation
- **FastAPI** for building the backend API
- **Uvicorn** as the ASGI server
- **ChromaDB** or **PostgreSQL + pgvector** for vector storage
- **Mermaid** for architecture & sequence diagrams in Markdown
- **Docker** & **docker-compose** for containerization

## System Architecture

```mermaid
flowchart LR
  subgraph Ingestion
    A["Raw Docs (PDF, HTML, Crawled)"] --> B["Text Reader (LlamaIndex Readers)"]
    B --> C["Chunker (RecursiveCharacterTextSplitter)"]
  end

  subgraph Embedding
    C --> D["GoogleGenAIEmbedding Task"]
    D -->|enqueue| CW1["Celery Worker: Embedding"]
  end

  subgraph Storage
    CW1 --> E["ChromaDB"]
  end

  subgraph Backend
    E --> F["Retrieval Engine (index.as_query_engine())"]
    F --> G["LLM Generation Task"]
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
    pip install fastapi uvicorn llama-index llama-index-embeddings-google-genai chromadb psycopg2-binary python-dotenv
    ```

2. **Data Ingestion & Chunking**
    ```python
    from llama_index import SimpleDirectoryReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = SimpleDirectoryReader("database/texts").load_data()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    ```

3. **Embedding & Indexing**
    ```python
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index import VectorStoreIndex, StorageContext

    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-005", api_key=os.getenv("GOOGLE_API_KEY"))
    storage_ctx = StorageContext.from_defaults(persist_dir="./.index_storage")
    index = VectorStoreIndex.from_documents(chunks, embed_model=embed_model, storage_context=storage_ctx)
    storage_ctx.persist()
    ```

4. **Backend API**
    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    from services.rag_service import answer_query

    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        top_k: int = 3

    @app.post("/query")
    def query_endpoint(req: QueryRequest):
        return {"answer": answer_query(req.query, req.top_k)}
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
