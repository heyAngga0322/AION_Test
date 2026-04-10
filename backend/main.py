from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import json
import re
from pypdf import PdfReader
from sqlmodel import Session, select
from typing import List, Any

from database import create_db_and_tables, get_session
from models import Document, Chunk, QueryRequest, QueryResponse
from nlpengine import qa_engine, process_text_into_chunks

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
    # Load existing chunks into BM25 engine
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

    # Re-build BM25 index
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

@app.post("/api/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest, session: Session = Depends(get_session)):
    results = qa_engine.ask(request.question)
    
    if not results:
        return QueryResponse(answer="I couldn't find an answer to that question in the provided documents.", sources=[])
    
    # Construct Extractive answer
    answer_parts = []
    sources = set()
    for chunk, score in results:
        doc = session.get(Document, chunk.document_id)
        if doc:
            answer_parts.append(chunk.text)
            sources.add(f"{doc.title}")
            
    final_answer = "\n\n".join(answer_parts)
    
    return QueryResponse(answer=final_answer, sources=list(sources))
