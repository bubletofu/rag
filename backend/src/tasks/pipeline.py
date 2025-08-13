#backend/src/tasks/pipeline.py

from celery import chain
from celery.utils.log import get_task_logger
from typing import Dict, Any, List
from langchain.schema import Document

from ..celery_app import celery_app
from ..services.ingest_utils import pdf_to_pages, chunk_pages, file_sha256
from ..services.rag_service import upsert_chunks 

log = get_task_logger(__name__)

@celery_app.task(bind=True)
def stage_raw_to_pages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = payload["path"]
    doc_id = payload.get("doc_id") or file_sha256(path)
    pages = pdf_to_pages(path)
    out = {"doc_id": doc_id, "path": path, "pages": [p.dict() for p in pages]}
    log.info(f"[{doc_id}] loaded pages={len(pages)}")
    return out

@celery_app.task(bind=True)
def stage_chunk_pages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    pages = [Document(**p) for p in payload["pages"]]
    chunks = chunk_pages(pages)
    out = {**payload, "chunks": [c.dict() for c in chunks]}
    log.info(f"[{payload['doc_id']}] chunks={len(chunks)}")
    return out

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 5})
def stage_upsert_chroma(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    chunks = [Document(**c) for c in payload["chunks"]]
    n = upsert_chunks(chunks)
    log.info(f"[{payload['doc_id']}] upserted={n}")
    return {"doc_id": payload["doc_id"], "path": payload["path"], "upserted": n}

def start_ingest_pipeline(path: str, doc_id: str | None = None) -> str:
    initial = {"path": path, "doc_id": doc_id}
    c = chain(
        stage_raw_to_pages.s(initial),
        stage_chunk_pages.s(),
        stage_upsert_chroma.s(),
    )
    result = c.apply_async()
    return result.id