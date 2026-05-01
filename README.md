# Multi-Modal Graph RAG System

This is a production-grade Multi-Modal Graph RAG application that supports file uploads (CSV, PDF, Images), extracts data, chunks it, and indexes it in both a Vector Database (Pinecone) and a Graph Database (Neo4j). It answers user questions using a hybrid retrieval mechanism powered by OpenAI.

## Technology Stack

- **Backend:** FastAPI, LangChain, Pinecone, Neo4j, OpenCV, PyMuPDF, Pandas
- **Frontend:** React (Vite), Framer Motion, Axios, Tailwind-like custom CSS
- **DevOps:** Docker, Docker Compose

## Prerequisites

- Node.js (v18+)
- Python 3.10+
- Docker and Docker Compose
- OpenAI API Key
- Pinecone API Key
- Neo4j Instance (or use the one provided via Docker Compose)

## Project Structure

```
graph-rag-app/
├── backend/                  # FastAPI Application
│   ├── core/                 # Configuration and settings
│   ├── models/               # Pydantic schemas
│   ├── services/             # Processing, vector DB, graph DB, and RAG pipeline
│   ├── main.py               # FastAPI entry point
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # Environment variables template
│   └── Dockerfile            # Backend Dockerfile
├── frontend/                 # React Application
│   ├── src/                  # React components and CSS
│   ├── package.json          # Node dependencies
│   └── Dockerfile            # Frontend Dockerfile
└── docker-compose.yml        # Docker composition setup
```

## Setup Instructions

### 1. Environment Variables

Navigate to the `backend` directory and create a `.env` file based on `.env.example`:

```bash
cd backend
cp .env.example .env
```

Fill in your actual API keys:
- `OPENAI_API_KEY`: Your OpenAI API Key
- `PINECONE_API_KEY`: Your Pinecone API Key
- `PINECONE_INDEX_NAME`: Your Pinecone Index Name (defaults to `graph-rag-index`)
- Keep Neo4j settings as they are if running via Docker.

### 2. Running with Docker Compose (Recommended)

To run the entire stack (Backend, Frontend, and Neo4j), navigate to the root directory and run:

```bash
docker-compose up --build
```

- The **Frontend** will be available at `http://localhost:5173` (or port 80 depending on Docker setup)
- The **Backend** will be available at `http://localhost:8000`
- The **Neo4j Browser** will be available at `http://localhost:7474` (login with `neo4j` / `password`)

### 3. Running Locally for Development

#### Backend
```bash
cd backend
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Open the application in your browser.
2. Drag and drop a file (CSV, PDF, or Image) into the upload zone.
3. Wait for the file to be processed, chunked, and inserted into Pinecone and Neo4j.
4. Ask questions in the chat interface. The system will retrieve context from both databases to form an accurate and explainable answer.
