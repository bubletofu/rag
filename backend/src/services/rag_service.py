# backend/src/services/rag_service.py


import os, shutil, hashlib
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma  
from langchain.schema import Document

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./database/pdfs")
PERSIST_DIR = os.getenv("INDEX_PERSIST_DIR", "./.index_storage")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_embed = None
_vs = None
_retriever = None

def _embeddings():
    global _embed
    if _embed is None:
        _embed = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
    return _embed

def get_vectorstore() -> Chroma:
    global _vs
    if _vs is None:
        _vs = Chroma(
            collection_name="rag",
            embedding_function=_embeddings(),
            persist_directory=PERSIST_DIR,
        )
    return _vs

def _build_retriever(k: int = 6):
    global _retriever
    _retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    )
    return _retriever

def get_retriever():
    if _retriever is None:
        _build_retriever()
    return _retriever

def _load_pdfs() -> List[Document]:
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("file_path") or ""
        d.metadata["source"] = os.path.basename(src)
    return docs

def _chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def _chunk_id(doc: Document) -> str:
    src = doc.metadata.get("source", "unknown")
    page = str(doc.metadata.get("page", ""))
    head = doc.page_content[:160]
    return hashlib.md5(f"{src}::{page}::{head}".encode("utf-8")).hexdigest()

def load_and_index_documents() -> int:
    docs = _load_pdfs()
    chunks = _chunk_docs(docs)
    print(f"[INFO] Loaded {len(docs)} pages → {len(chunks)} chunks")
    vs = get_vectorstore()
    # (Optional) add your own dedupe via cid in metadata
    for c in chunks:
        c.metadata["cid"] = _chunk_id(c)
    vs.add_documents(chunks)     # auto-persist in Chroma >= 0.4
    _build_retriever()
    return len(chunks)

def rebuild_index() -> int:
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    global _vs, _retriever
    _vs = None
    _retriever = None
    return load_and_index_documents()

SYS_PROMPT = (
    "You are a precise assistant. Answer using ONLY the provided context. "
    "If the answer is not in the context, say you don't know. "
    'Cite sources like [file.pdf p.X].'
)

def _format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        parts.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)

# expose a simple upsert helper for Celery
def upsert_chunks(chunks):
    vs = get_vectorstore()  
    vs.add_documents(chunks)
    return len(chunks)

async def answer_query(query: str, k: int = 6,
                       filters: Optional[Dict[str, Any]] = None,
                       score_threshold: Optional[float] = None) -> Dict[str, Any]:
    vs = get_vectorstore()
    docs_scores: List[Tuple[Document, float]] = vs.similarity_search_with_score(
        query, k=max(k, 6), filter=filters
    )
    if score_threshold is not None:
        # For cosine distance, lower is better → keep <= threshold
        docs_scores = [ds for ds in docs_scores if ds[1] <= score_threshold]

    docs = []
    seen = set()
    for d, _ in docs_scores:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            docs.append(d)
        if len(docs) >= k:
            break

    context = _format_context(docs)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0,
                     api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
    ]
    resp = await llm.ainvoke(messages)

    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    retrieval = [{"source": ds[0].metadata.get("source"),
                  "page": ds[0].metadata.get("page"),
                  "score": ds[1]} for ds in docs_scores[:k]]

    return {"answer": resp.content, "sources": sources, "retrieval": retrieval}