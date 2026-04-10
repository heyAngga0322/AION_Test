import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import Chunk

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

qa_engine = QAEngine()

def process_text_into_chunks(text: str) -> List[str]:
    # Better chunking: group lines into semantic paragraphs
    text = normalize_text(text)
    lines = text.split('\n')
    
    chunks = []
    current_chunk = []
    
    # Pattern to detect TOC lines (e.g., "1.1.1 Title .......... 5")
    toc_pattern = re.compile(r'\.{3,}')
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # Empty line = paragraph break
            if current_chunk:
                chunk_text = normalize_chunk_text(' '.join(current_chunk))
                if len(chunk_text) > 50 and not toc_pattern.search(chunk_text):
                    chunks.append(chunk_text)
                current_chunk = []
            continue
        
        # Skip TOC lines
        if toc_pattern.search(stripped):
            continue
            
        # Skip standalone headers (short lines with mostly numbers/dots)
        if len(stripped) < 40 and re.match(r'^[\d\.\s]+[A-Za-z\s]+$', stripped):
            if current_chunk:
                chunk_text = normalize_chunk_text(' '.join(current_chunk))
                if len(chunk_text) > 50:
                    chunks.append(chunk_text)
                current_chunk = []
            continue
            
        current_chunk.append(stripped)
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = normalize_chunk_text(' '.join(current_chunk))
        if len(chunk_text) > 50 and not toc_pattern.search(chunk_text):
            chunks.append(chunk_text)
    
    return chunks if chunks else [normalize_chunk_text(text)]
