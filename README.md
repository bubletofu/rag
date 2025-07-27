RAG Project

This repository contains a full-stack Retrieval-Augmented Generation (RAG) application built with:
	•	Backend: FastAPI + LlamaIndex
	•	Frontend: (to be added)
	•	Database: ChromaDB for vector storage

⸻

📦 Setup the Environment
	1.	Navigate to the backend directory

cd backend


	2.	Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate


	3.	Define dependencies
Create a requirements.txt with the following content:

fastapi
uvicorn[standard]
llama-index
openai
python-dotenv
chromadb


	4.	Install dependencies

pip install -r requirements.txt


	5.	Configure environment variables
Copy the example and set your OpenAI key and optional index paths:

cp .env.example .env
# Then edit .env:
# OPENAI_API_KEY=sk-...
# DATA_DIR=../database/texts
# INDEX_PERSIST_DIR=./.index_storage



⸻

🚀 Running the Backend

Start the FastAPI server:

uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

The RAG index will build automatically on startup (first run) and persist for subsequent calls.

⸻

🛠️ Next Steps
	•	Add your .txt or .md documents under database/texts/.
	•	Build or load the vector index by restarting the server.
	•	Scaffold the frontend to call POST /query and display RAG responses.
	•	Dockerize services with docker-compose.yml.