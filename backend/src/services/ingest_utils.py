#backend/src/services/ingest_utils.py

import os, hashlib
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def pdf_to_pages(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    pages = loader.load()
    src = os.path.basename(path)
    for d in pages:
        d.metadata["source"] = src
    return pages

def chunk_pages(pages: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)