import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models import Chunk

class QAEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
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

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.chunk_vectors).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.05: # Similarity threshold
                results.append((self.chunks_data[idx], score))
                
        return results

qa_engine = QAEngine()

def process_text_into_chunks(text: str) -> List[str]:
    # Better chunking: group lines into semantic paragraphs
    text = text.replace('\r', '\n')
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
                chunk_text = ' '.join(current_chunk).strip()
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
                chunk_text = ' '.join(current_chunk).strip()
                if len(chunk_text) > 50:
                    chunks.append(chunk_text)
                current_chunk = []
            continue
            
        current_chunk.append(stripped)
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk).strip()
        if len(chunk_text) > 50 and not toc_pattern.search(chunk_text):
            chunks.append(chunk_text)
    
    return chunks if chunks else [text.strip()]
