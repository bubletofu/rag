#backend/src/main.py

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from src.services.rag_service import load_and_index_documents
import os
from dotenv import load_dotenv
import shutil

load_dotenv() #tét git

app = FastAPI()

vectorstore = load_and_index_documents(use_openai=True)
llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    result = qa_chain(req.query)
    return {
        "answer": result['result'],
        "sources": [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page_content": doc.page_content
            }
            for doc in result['source_documents']
        ]
    }

@app.post("/load-document")
async def load_document(file: UploadFile = File(...)):
     
    ext = file.filename.split(".")[-1]
    target_dir = os.getenv("DATA_DIR") if ext == "pdf" else os.getenv("DATA_DIR").replace("pdfs", "web_pages")
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    global vectorstore, retriever, qa_chain
    vectorstore = load_and_index_documents(use_openai=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return {"message": f"Document {file.filename} loaded and indexed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
