# backend/src/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os, shutil, glob

from .tasks.pipeline import start_ingest_pipeline
from .celery_app import celery_app


from .services.rag_service import (
    # load_and_index_documents,
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

# @app.post("/load-document")
# async def load_document(file: UploadFile = File(...)):
#     data_dir = os.getenv("DATA_DIR", "./database/pdfs")
#     os.makedirs(data_dir, exist_ok=True)
#     dst = os.path.join(data_dir, file.filename)
#     with open(dst, "wb") as buf:
#         shutil.copyfileobj(file.file, buf)
#     n = load_and_index_documents()
#     return {"message": f"Loaded '{file.filename}'", "chunks_indexed": n}

@app.post("/rebuild-index")
async def rebuild():
    n = rebuild_index()
    return {"status": "ok", "chunks": n}

@app.get("/sources")
async def list_sources():
    root = os.getenv("DATA_DIR", "./database/pdfs")
    files = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*.pdf"))]
    return {"sources": files}

# Upload → enqueue ingestion pipeline (async)
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data_dir = os.getenv("DATA_DIR", "./database/pdfs")
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, file.filename)
    with open(dst, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    task_id = start_ingest_pipeline(dst)
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"task_id": task_id, "file": file.filename},
        headers={"Location": f"/ingest/{task_id}/status"},
    )

# enqueue ingestion for an existing filesystem path
class IngestExistingRequest(BaseModel):
    path: str
    doc_id: str | None = None

@app.post("/ingest-existing")
async def ingest_existing(req: IngestExistingRequest):
    task_id = start_ingest_pipeline(req.path, req.doc_id)
    return {"task_id": task_id, "path": req.path}

# Poll Celery task status
@app.get("/ingest/{task_id}/status")
def ingest_status(task_id: str):
    res = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": res.state,                  # PENDING/STARTED/SUCCESS/FAILURE/RETRY
        "result": res.result if res.successful() else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.main:app", host="0.0.0.0", port=8000, reload=True)