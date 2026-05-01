import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader, Database } from 'lucide-react';
import axios from 'axios';
import { motion } from 'framer-motion';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your Multi-Modal Graph RAG Assistant. Ask me anything about your uploaded data.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await axios.post('http://localhost:8000/api/query', { query: userMessage.content });
      
      const assistantMessage = {
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources,
        graphContext: res.data.graph_context
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error while processing your request.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="panel" style={{ height: '100%' }}>
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <motion.div 
              key={idx} 
              className={`message ${msg.role}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div style={{ marginBottom: '0.5rem', fontWeight: msg.role === 'assistant' ? 300 : 500 }}>
                {msg.content}
              </div>
              
              {msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Database size={12} /> Sources used:
                  </span>
                  <div className="source-badges">
                    {msg.sources.map((src, i) => (
                      <span key={i} className="source-badge">
                        {src.metadata?.source || 'Unknown'} (Chunk {src.id})
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
          {isLoading && (
            <div className="message assistant">
              <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}>
                Thinking...
              </motion.div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about your data..."
            disabled={isLoading}
          />
          <button onClick={handleSend} disabled={isLoading || !input.trim()}>
            {isLoading ? <Loader size={20} className="spin" /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
