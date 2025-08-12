# backend/src/main.py
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os, shutil, glob

from .services.rag_service import (
    load_and_index_documents,
    rebuild_index,
    answer_query,
)

load_dotenv()

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", 
                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 6
    filters: dict | None = None
    score_threshold: float | None = None

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    return await answer_query(
        query=req.query,
        k=req.top_k,
        filters=req.filters,
        score_threshold=req.score_threshold,
    )

@app.post("/load-document")
async def load_document(file: UploadFile = File(...)):
    data_dir = os.getenv("DATA_DIR", "./database/pdfs")
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, file.filename)
    with open(dst, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    n = load_and_index_documents()
    return {"message": f"Loaded '{file.filename}'", "chunks_indexed": n}

@app.post("/rebuild-index")
async def rebuild():
    n = rebuild_index()
    return {"status": "ok", "chunks": n}

@app.get("/sources")
async def list_sources():
    root = os.getenv("DATA_DIR", "./database/pdfs")
    files = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*.pdf"))]
    return {"sources": files}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.main:app", host="0.0.0.0", port=8000, reload=True)