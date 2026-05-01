import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import axios from 'axios';
import { motion } from 'framer-motion';

const FileUpload = () => {
  const [status, setStatus] = useState({ type: '', message: '' });
  const [isLoading, setIsLoading] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setIsLoading(true);
    setStatus({ type: '', message: 'Uploading and processing...' });

    try {
      const res = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setStatus({ 
        type: 'success', 
        message: `Success! Extracted ${res.data.chunks_processed} chunks and ${res.data.entities_extracted} graph entities.` 
      });
    } catch (err) {
      console.error(err);
      setStatus({ 
        type: 'error', 
        message: err.response?.data?.detail || 'An error occurred during upload.' 
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  return (
    <div className="panel">
      <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem' }}>Data Sources</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        Upload CSV, PDF, or Images to build the knowledge graph.
      </p>

      <div 
        {...getRootProps()} 
        className={`upload-zone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        {isLoading ? (
          <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: "linear" }}>
            <Loader size={48} className="upload-icon" />
          </motion.div>
        ) : (
          <UploadCloud size={48} className="upload-icon" />
        )}
        
        {isDragActive ? (
          <p>Drop the file here ...</p>
        ) : (
          <p>Drag 'n' drop a file here, or click to select</p>
        )}
      </div>

      {status.message && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="status-message"
          style={{ 
            background: status.type === 'error' ? 'rgba(239, 68, 68, 0.1)' : '',
            color: status.type === 'error' ? '#ef4444' : '',
            borderColor: status.type === 'error' ? 'rgba(239, 68, 68, 0.2)' : ''
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {status.type === 'error' ? <AlertCircle size={16} /> : <CheckCircle size={16} />}
            <span>{status.message}</span>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
