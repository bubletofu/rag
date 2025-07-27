import os
import shutil
import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.services.rag_service import PERSIST_DIR, build_or_load_index
from src.api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_index_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_PERSIST_DIR", str(tmp_path / "index"))
    yield
    if (tmp_path / "index").exists():
        shutil.rmtree(tmp_path / "index")

# def test_health_endpoint():
#     resp = client.get("/health")
#     assert resp.status_code == 200
#     assert resp.json() == {"status": "ok"}

def test_query_endpoint():
    build_or_load_index()
    payload = {"query": "What is the main topic?", "top_k": 1}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert len(data["answer"]) > 1