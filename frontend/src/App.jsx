import React from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="app-container">
      <header className="header">
        <h1>Graph RAG Nexus</h1>
        <p>Multi-Modal · FAISS · Reranker · Phi-3 Mini (Ollama) · Fully Offline</p>
      </header>

      <main style={{ display: 'contents' }}>
        <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <FileUpload />

          <div className="panel" style={{ flex: 1 }}>
            <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>System Status</h3>
            <ul style={{ listStyle: 'none', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Vector DB (FAISS):</span>
                <span style={{ color: 'var(--success-color)' }}>Offline ✓</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Graph DB (Neo4j):</span>
                <span style={{ color: 'var(--success-color)' }}>bolt://localhost:7687</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>LLM (Ollama Phi-3):</span>
                <span style={{ color: 'var(--success-color)' }}>Local ✓</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Embeddings (CLIP + MiniLM):</span>
                <span style={{ color: 'var(--success-color)' }}>CPU ✓</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Reranker (CrossEncoder):</span>
                <span style={{ color: 'var(--success-color)' }}>CPU ✓</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="main-content">
          <ChatInterface />
        </div>
      </main>
    </div>
  );
}

export default App;
