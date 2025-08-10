import os
from dotenv import load_dotenv
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def pdf_to_txt(pdf_path, txt_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"[✓] Saved extracted text to {txt_path}")
    return txt_path

def read_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

# === Paths ===
PDF_PATH = "./database/pdfs/A comparison of static, dynamic, and hybrid analysis for malware detection.pdf"
TXT_PATH = "./database/pdfs/output.txt"

# === Step 1: Extract text from PDF ===
pdf_to_txt(PDF_PATH, TXT_PATH)

# === Step 2: Read text ===
full_text = read_txt(TXT_PATH)

# === Step 3: Chunk text ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([full_text])

# === Step 4: Print chunks ===
print(f"Total chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk.page_content[:500])
    print()