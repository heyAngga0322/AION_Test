import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from models import Chunk

class QAEngine:
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus: List[List[str]] = []
        self.chunks_data: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Unicode-friendly tokenization."""
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def load_chunks(self, chunks: List[Chunk]):
        self.chunks_data = chunks
        if not chunks:
            self.bm25 = None
            self.tokenized_corpus = []
            return

        # Tokenize each chunk for BM25
        self.tokenized_corpus = [self._tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def ask(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        if self.bm25 is None or not self.chunks_data:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        k = max(1, min(top_k, len(self.chunks_data)))
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0.0:
                results.append((self.chunks_data[idx], float(score)))

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
