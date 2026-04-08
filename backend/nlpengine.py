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
    # Simple chunking by paragraph or sentences based on newlines.
    text = text.replace('\r', '\n')
    raw_chunks = re.split(r'\n+', text)
    processed = []
    for chunk in raw_chunks:
        cl = chunk.strip()
        if len(cl) > 20: # skip very small noisy lines
            processed.append(cl)
    return processed
