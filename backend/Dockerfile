
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire backend source code into container
COPY backend ./backend
COPY database ./database

# Set environment variables (optional, helpful for local dev)
ENV PYTHONUNBUFFERED=1

# Run the FastAPI app via Python so it enters through __main__
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]


