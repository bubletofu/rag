# rag/backend/src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.services.rag_service import answer_query, build_index

app = FastAPI(title="RAG Service")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.on_event("startup")
async def startup_event():
    build_index()

@app.post("/query")
def query(request: QueryRequest):
    try:
        result = answer_query(request.query, k=request.top_k)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))