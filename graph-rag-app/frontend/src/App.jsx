import React from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';

function App() {
  return (
    <div className="app-container">
      <header className="header">
        <h1>Graph RAG Nexus</h1>
        <p>Enterprise-Grade Multi-Modal Knowledge Retrieval</p>
      </header>

      <main style={{ display: 'contents' }}>
        <div className="sidebar" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <FileUpload />
          
          <div className="panel" style={{ flex: 1 }}>
            <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>System Status</h3>
            <ul style={{ listStyle: 'none', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Vector DB (Pinecone):</span>
                <span style={{ color: 'var(--success-color)' }}>Online</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Graph DB (Neo4j):</span>
                <span style={{ color: 'var(--success-color)' }}>Online</span>
              </li>
              <li style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>LLM Engine (Gemini):</span>
                <span style={{ color: 'var(--success-color)' }}>Active</span>
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
