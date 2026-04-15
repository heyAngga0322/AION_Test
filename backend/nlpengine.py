import re
import os
from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer

from models import Chunk

# Download required NLTK data on first import
# Use a custom directory to avoid permission issues
def ensure_nltk_data():
    nltk_data_dir = os.path.expanduser("~/nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add custom path to NLTK's data path first
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    # Download punkt tokenizer data
    try:
        nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    
    # Also download punkt_tab for NLTK 3.9+ compatibility
    try:
        nltk.data.find('tokenizers/punkt_tab', paths=[nltk_data_dir])
    except LookupError:
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)

ensure_nltk_data()

def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    # Fix common PDF extraction artifacts like "dif- ficult" / "experi- ence"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text, flags=re.UNICODE)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_chunk_text(text: str) -> str:
    text = normalize_text(text)
    # Remove leading section numbering (e.g., "1.1.1", "2.3", "10.4.2")
    text = re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", text)
    return text.strip()

class QAEngine:
    def __init__(self):
        # Initialize Sentence Transformer model for semantic embeddings
        # Model: all-MiniLM-L6-v2 creates 384-dimensional embeddings
        # This replaces TF-IDF for better semantic understanding
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.chunks_data = []

    def load_chunks(self, chunks: List[Chunk]):
        """RAG Process Step 1: Document Ingestion & Embedding
        
        This method encodes document chunks into semantic embeddings.
        These embeddings will be used for similarity search during retrieval.
        """
        self.chunks_data = chunks
        if not chunks:
            self.embeddings = None
            return
        
        # Convert text chunks to semantic embeddings
        texts = [chunk.text for chunk in chunks]
        # encode() creates normalized embeddings for better similarity calculation
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def ask(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """RAG Process Step 2: Retrieval
        
        This method performs semantic similarity search to find the most relevant
        document chunks for the given query using cosine similarity between embeddings.
        """
        if self.embeddings is None or not self.chunks_data:
            return []

        query = (query or "").strip()
        if not query:
            return []

        # Encode query to embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Calculate cosine similarity between query and all document embeddings
        # np.dot() with normalized embeddings gives cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top-k most similar chunks
        k = max(1, min(top_k, len(self.chunks_data)))
        top_indices = similarities.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            score = similarities[idx]
            results.append((self.chunks_data[idx], float(score)))

        return results

    def ask_all_relevant(self, query: str, threshold: float = 0.3) -> List[Tuple[Chunk, float]]:
        """RAG Process Step 2: Enhanced Retrieval
        
        Retrieves ALL chunks above a relevance threshold for comprehensive answer synthesis.
        Uses semantic embeddings for better understanding of query intent.
        """
        if self.embeddings is None or not self.chunks_data:
            return []

        query = (query or "").strip()
        if not query:
            return []

        # Encode query to embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Calculate semantic similarities
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get all indices above threshold, sorted by score descending
        # Threshold 0.3 for semantic embeddings (higher than TF-IDF due to normalization)
        relevant_indices = [i for i, s in enumerate(similarities) if s >= threshold]
        relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        
        results = []
        for idx in relevant_indices:
            results.append((self.chunks_data[idx], float(similarities[idx])))

        return results

qa_engine = QAEngine()

def process_text_into_chunks(text: str) -> List[str]:
    # Sentence-level chunking using NLTK for precise answers
    text = normalize_text(text)
    
    # Use NLTK to split into sentences
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    min_chunk_length = 80  # Minimum characters per chunk
    max_chunk_length = 500  # Maximum characters per chunk
    
    # Pattern to detect TOC lines and headers
    toc_pattern = re.compile(r'\.{3,}')
    header_pattern = re.compile(r'^[\d\.\s]+[A-Za-z\s]+$')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Skip TOC lines
        if toc_pattern.search(sentence):
            continue
        
        # Skip standalone headers (short lines with mostly numbers/dots)
        if len(sentence) < 40 and header_pattern.match(sentence):
            continue
        
        # Clean the sentence
        sentence = normalize_chunk_text(sentence)
        if len(sentence) < 20:  # Skip very short fragments
            continue
        
        current_chunk.append(sentence)
        current_text = ' '.join(current_chunk)
        
        # Check if we should finalize this chunk
        if len(current_text) >= min_chunk_length:
            chunks.append(current_text)
            current_chunk = []
        elif len(current_text) > max_chunk_length:
            # Force split if too long
            chunks.append(current_text)
            current_chunk = []
    
    # Add remaining sentences as final chunk
    if current_chunk:
        final_chunk = ' '.join(current_chunk)
        if len(final_chunk) >= 30:  # Lower threshold for final chunk
            chunks.append(final_chunk)
    
    return chunks if chunks else [normalize_chunk_text(text)]
