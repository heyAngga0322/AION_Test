from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import json
import re
from pypdf import PdfReader
from sqlmodel import Session, select
from typing import List, Any, Dict, Tuple

from database import create_db_and_tables, get_session
from models import Document, Chunk, QueryRequest, QueryResponse
from nlpengine import qa_engine, process_text_into_chunks, normalize_chunk_text

def shorten_answer(text: str, max_sentences: int = 3, max_chars: int = 900) -> str:
    text = text.strip()
    if not text:
        return text

    # Split into sentences (simple heuristic; keeps extractive content).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    shortened = " ".join(s for s in sentences[:max_sentences] if s).strip()
    if not shortened:
        shortened = text

    if len(shortened) > max_chars:
        shortened = shortened[:max_chars].rsplit(" ", 1)[0].rstrip()
        shortened = shortened + "…"

    return shortened

app = FastAPI(title="QA Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Load existing chunks into TF-IDF engine
    from database import engine
    with Session(engine) as session:
        chunks = session.exec(select(Chunk)).all()
        qa_engine.load_chunks(list(chunks))

@app.post("/api/documents")
async def upload_document(
    title: str = Form(...),
    file: UploadFile = File(None),
    text_content: str = Form(None),
    session: Session = Depends(get_session)
):
    content = ""
    if file:
        content_bytes = await file.read()
        if file.filename and file.filename.lower().endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(content_bytes))
            content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    content += extracted + "\n"
        elif file.filename and file.filename.lower().endswith('.json'):
            json_content = content_bytes.decode('utf-8', errors='replace')
            content = extract_text_from_json(json_content)
        else:
            content = content_bytes.decode('utf-8', errors='replace')
    elif text_content:
        content = text_content
    else:
        raise HTTPException(status_code=400, detail="Must provide file or text_content")

    # Save document
    doc = Document(title=title, content=content)
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Document Chunking
    chunks_text = process_text_into_chunks(content)
    new_chunks = []
    for i, ctx in enumerate(chunks_text):
        chunk = Chunk(document_id=doc.id, text=ctx, chunk_index=i)
        session.add(chunk)
        new_chunks.append(chunk)
    
    session.commit()

    # Re-build TF-IDF index
    all_chunks = session.exec(select(Chunk)).all()
    qa_engine.load_chunks(list(all_chunks))

    return {"message": f"Document '{title}' uploaded and split into {len(new_chunks)} chunks."}

def clean_text(text: str) -> str:
    """Clean text: lowercase, strip HTML tags, remove special characters."""
    # Convert to lowercase
    text = text.lower()
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove special characters but keep alphanumeric, spaces, and basic punctuation
    text = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_json(json_str: str) -> str:
    """Recursively extract and clean all string values from JSON into flat text."""
    try:
        data = json.loads(json_str)
        texts = []
        
        def extract(obj: Any):
            if isinstance(obj, str):
                # Clean the text before adding
                cleaned = clean_text(obj)
                if cleaned and len(cleaned) > 2:  # Skip very short strings
                    texts.append(cleaned)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item)
        
        extract(data)
        return "\n".join(texts)
    except json.JSONDecodeError:
        # Try to clean raw string even if not valid JSON
        return clean_text(json_str)

def synthesize_answer(question: str, chunks_with_scores: List[Tuple[Chunk, float]], session: Session) -> Tuple[str, List[str]]:
    """Synthesize a comprehensive answer from multiple relevant chunks.
    
    This function combines information from all relevant chunks to create
    a coherent answer that reads all available resources and concludes.
    """
    if not chunks_with_scores:
        return "I couldn't find an answer to that question in the provided documents.", []
    
    # Group chunks by document for better organization
    doc_chunks: Dict[int, List[Tuple[str, float]]] = {}
    sources = set()
    
    for chunk, score in chunks_with_scores:
        doc = session.get(Document, chunk.document_id)
        if not doc:
            continue
        
        cleaned = normalize_chunk_text(chunk.text)
        if not cleaned or len(cleaned) < 10:
            continue
            
        if doc.id not in doc_chunks:
            doc_chunks[doc.id] = []
        doc_chunks[doc.id].append((cleaned, score))
        sources.add(doc.title)
    
    if not doc_chunks:
        return "I couldn't find relevant information to answer this question.", []
    
    # Build synthesized answer
    answer_parts = []
    
    # If only one document, combine all relevant chunks
    if len(doc_chunks) == 1:
        doc_id = list(doc_chunks.keys())[0]
        chunks = doc_chunks[doc_id]
        
        if len(chunks) == 1:
            # Single best answer
            answer_parts.append(chunks[0][0])
        else:
            # Multiple relevant passages - combine them
            answer_parts.append("Based on the document, here are the relevant details:")
            for i, (text, score) in enumerate(chunks[:5], 1):  # Top 5 from single doc
                if i == 1:
                    answer_parts.append(text)  # First one is the main answer
                else:
                    answer_parts.append(f"Additionally: {text}")
    else:
        # Multiple documents - synthesize across sources
        answer_parts.append("Based on information from multiple sources:")
        
        for doc_id, chunks in doc_chunks.items():
            doc = session.get(Document, doc_id)
            if not doc:
                continue
            
            # Get the best chunk from this document
            best_text, best_score = chunks[0]
            answer_parts.append(f"\nFrom '{doc.title}': {best_text}")
            
            # Add secondary info if available and relevant
            if len(chunks) > 1 and chunks[1][1] > 0.15:
                answer_parts.append(f"  Also noted: {chunks[1][0]}")
    
    final_answer = "\n\n".join(answer_parts)
    return final_answer, list(sources)

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, session: Session = Depends(get_session)):
    # Retrieve ALL relevant chunks above threshold
    results = qa_engine.ask_all_relevant(request.question, threshold=0.05)
    
    # Synthesize comprehensive answer from all relevant resources
    final_answer, sources = synthesize_answer(request.question, results, session)
    
    return QueryResponse(answer=final_answer, sources=sources)
