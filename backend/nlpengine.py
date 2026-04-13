import re
import os
from typing import List, Tuple

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = None
        self.chunks_data: List[Chunk] = []

    def load_chunks(self, chunks: List[Chunk]):
        self.chunks_data = chunks
        if not chunks:
            self.chunk_vectors = None
            return

        texts = [chunk.text for chunk in chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def ask(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.chunk_vectors is None or not self.chunks_data:
            return []

        query = (query or "").strip()
        if not query:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors).flatten()
        k = max(1, min(top_k, len(self.chunks_data)))
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            score = scores[idx]
            results.append((self.chunks_data[idx], float(score)))

        return results

    def ask_all_relevant(self, query: str, threshold: float = 0.1) -> List[Tuple[Chunk, float]]:
        """Retrieve all chunks above a relevance threshold, sorted by score."""
        if self.chunk_vectors is None or not self.chunks_data:
            return []

        query = (query or "").strip()
        if not query:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors).flatten()
        
        # Get all indices above threshold, sorted by score descending
        relevant_indices = [i for i, s in enumerate(scores) if s >= threshold]
        relevant_indices.sort(key=lambda i: scores[i], reverse=True)
        
        results = []
        for idx in relevant_indices:
            results.append((self.chunks_data[idx], float(scores[idx])))

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
