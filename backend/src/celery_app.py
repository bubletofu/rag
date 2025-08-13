#backend/src/celery_app.py

from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "rag_ingest",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.src.tasks.pipeline"],  
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_time_limit=60 * 30,
    worker_prefetch_multiplier=1,
)