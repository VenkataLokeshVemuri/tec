import os
import cv2
import pandas as pd
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_csv(filepath: str) -> list[str]:
    df = pd.read_csv(filepath)
    # Convert CSV to JSON-like strings per row for better context
    records = df.to_dict(orient='records')
    texts = [str(record) for record in records]
    return chunk_texts(texts)

def process_pdf(filepath: str) -> list[str]:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    
    return chunk_text(text)

def process_image(filepath: str) -> list[str]:
    # Basic OpenCV preprocessing as requested
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Since OCR is not strictly required and we want it to run on a normal laptop easily,
    # we simulate extraction or just return metadata. In a real scenario, use pytesseract.
    height, width = img.shape[:2]
    simulated_text = f"Image metadata - Width: {width}, Height: {height}. Image was preprocessed with Grayscale and GaussianBlur. Content extraction requires an OCR model."
    return [simulated_text]

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def chunk_texts(texts: list[str]) -> list[str]:
    # If already a list of strings (like from CSV rows), we can still chunk them if they are too long
    # Or just return them if they are short enough. We'll join and split to be safe.
    combined = "\n".join(texts)
    return chunk_text(combined)

def process_file(filepath: str, filename: str) -> list[str]:
    ext = filename.split('.')[-1].lower()
    if ext == 'csv':
        return process_csv(filepath)
    elif ext == 'pdf':
        return process_pdf(filepath)
    elif ext in ['png', 'jpg', 'jpeg']:
        return process_image(filepath)
    else:
        # Fallback to plain text read
        with open(filepath, 'r', encoding='utf-8') as f:
            return chunk_text(f.read())
