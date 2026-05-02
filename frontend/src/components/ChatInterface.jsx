import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader, Database, Image, FileText } from 'lucide-react';
import axios from 'axios';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ChatInterface = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I am your Multi-Modal Graph RAG Assistant powered by Phi-3 Mini (Ollama). ' +
        'Upload documents or images, then ask me anything about your data.'
    }
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
      const res = await axios.post('http://localhost:8000/api/query', {
        query: userMessage.content
      });

      const assistantMessage = {
        role:         'assistant',
        content:      res.data.answer,
        textSources:  res.data.text_sources  || res.data.sources || [],
        imageSources: res.data.image_sources || [],
        graphContext: res.data.graph_context  || []
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        {
          role:    'assistant',
          content: 'Sorry, I encountered an error while processing your request.'
        }
      ]);
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
              <div style={{ marginBottom: '0.5rem', fontWeight: msg.role === 'assistant' ? 300 : 500 }} className="markdown-body">
                {msg.role === 'assistant' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code({node, inline, className, children, ...props}) {
                        const match = /language-(\w+)/.exec(className || '')
                        return !inline && match ? (
                          <SyntaxHighlighter
                            style={vscDarkPlus}
                            language={match[1]}
                            PreTag="div"
                            {...props}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        ) : (
                          <code className={className} style={{background: 'rgba(255,255,255,0.1)', padding: '2px 4px', borderRadius: '4px'}} {...props}>
                            {children}
                          </code>
                        )
                      }
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                ) : (
                  msg.content
                )}
              </div>

              {/* Text sources with rerank scores */}
              {msg.textSources && msg.textSources.length > 0 && (
                <div style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <FileText size={12} /> Text sources (reranked):
                  </span>
                  <div className="source-badges">
                    {msg.textSources.map((src, i) => (
                      <span key={i} className="source-badge" title={src.text?.slice(0, 120)}>
                        {src.metadata?.source || 'Unknown'} · chunk {src.id}
                        {src.metadata?.rerank_score != null &&
                          ` · score: ${Number(src.metadata.rerank_score).toFixed(2)}`}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Image sources with CLIP scores */}
              {msg.imageSources && msg.imageSources.length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Image size={12} /> Image sources (CLIP):
                  </span>
                  <div className="source-badges">
                    {msg.imageSources.map((src, i) => (
                      <span key={i} className="source-badge" style={{ background: 'rgba(139,92,246,0.15)' }}>
                        {src.metadata?.file_path?.split(/[\\/]/).pop() || `Image ${i + 1}`}
                        {src.metadata?.score != null &&
                          ` · sim: ${Number(src.metadata.score).toFixed(2)}`}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Graph context badge */}
              {msg.graphContext && msg.graphContext.length > 0 && msg.graphContext[0]?.graph_summary && (
                <div style={{ marginTop: '0.5rem' }}>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Database size={11} /> Graph context included
                  </span>
                </div>
              )}
            </motion.div>
          ))}

          {isLoading && (
            <div className="message assistant">
              <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 1.5 }}>
                Thinking... (Phi-3 Mini via Ollama)
              </motion.div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <input
            type="text"
            id="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask about your data..."
            disabled={isLoading}
          />
          <button id="chat-send-btn" onClick={handleSend} disabled={isLoading || !input.trim()}>
            {isLoading ? <Loader size={20} className="spin" /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
