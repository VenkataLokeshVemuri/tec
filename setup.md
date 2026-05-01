# Multi-Modal Graph RAG Setup Guide

This guide will walk you through the complete setup of the Multi-Modal Graph RAG system, including provisioning API keys, configuring the databases, and running the services using Docker and local servers.

## 1. Prerequisites

Before starting, ensure you have the following installed on your machine:
- **Docker & Docker Compose** (for running Neo4j)
- **Python 3.10+** (for the FastAPI backend)
- **Node.js 18+** (for the React frontend)
- **Git**

## 2. Acquiring API Keys

You will need two API keys for this project:

### Google Gemini API Key
This project uses Google's Gemini models for embedding, extraction, and generation.
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click on **Get API key** in the left sidebar and create a new key.
4. Save this key; you will need it later.

### Pinecone API Key
Pinecone is used as the Vector Database for semantic search.
1. Go to [Pinecone](https://app.pinecone.io/) and create a free account.
2. Navigate to **API Keys** on the left menu and copy your default key.
3. Keep track of the default index name we will use: `graph-rag-gemini`. (The backend will automatically create this index with a dimension of `3072` if it doesn't exist).

## 3. Environment Variables Configuration

In the `backend` directory, create a `.env` file (you can copy `.env.example` if it exists) and fill it with your credentials:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=graph-rag-gemini

# Neo4j Graph DB Configuration
# Use bolt://localhost:7687 if running the backend locally.
# Use bolt://neo4j:7687 if running the backend inside docker.
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=madhukar
```

## 4. Setting up Neo4j via Docker

We use Docker to run the Neo4j graph database locally with the **APOC plugin** enabled (which is required by LangChain).

1. Open your terminal in the root directory of the project (`d:\tec\graph-rag-app`).
2. Start the Neo4j container in the background:
   ```bash
   docker-compose up -d neo4j
   ```
3. Docker will pull the `neo4j:5.12.0` image, enable the APOC plugin via environment variables, and map the necessary ports (`7474` and `7687`).

### Accessing the Neo4j Browser
Once the container is running, you can view your graph database:
1. Open your web browser and navigate to [http://localhost:7474](http://localhost:7474).
2. Connect with the following details:
   - **Connect URL:** `bolt://localhost:7687`
   - **Username:** `neo4j`
   - **Password:** `madhukar`

## 5. Running the Application Locally

Instead of running everything in Docker, you can run the frontend and backend locally for easier development and hot-reloading.

### Backend (FastAPI)
1. Open a terminal and navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
   **Backend Endpoints:** 
   - API Base: `http://127.0.0.1:8000`
   - Interactive API Docs (Swagger): `http://127.0.0.1:8000/docs`

### Frontend (React/Vite)
1. Open a new terminal and navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   **Frontend Endpoint:** `http://localhost:5173`

## 6. Using the Application

1. Open the frontend at `http://localhost:5173`.
2. Drag and drop a document (PDF, CSV, or Text) into the upload zone.
3. Wait for the success message. The backend will parse the file, generate embeddings via Gemini, upsert them into Pinecone, and extract entities/relationships into Neo4j.
4. Use the chat interface to query your data. The RAG pipeline will fetch semantic matches from Pinecone and relationship contexts from Neo4j to formulate a comprehensive answer!
