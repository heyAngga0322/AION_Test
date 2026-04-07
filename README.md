# Question Answering Service

A Python-based Question Answering (QA) service that allows users to upload documents and ask questions based on that content. The system generates answers strictly using internal logic without relying on external AI APIs.

## Features

- **Document Upload**: Upload text or JSON content
- **Smart Text Processing**: Automatic chunking with overlaps, preprocessing, and cleaning
- **Multiple Retrieval Methods**: Both TF-IDF similarity and keyword matching
- **Answer Generation**: Constructs answers from retrieved content with source references
- **Confidence Scoring**: Provides confidence metrics for generated answers
- **Document Management**: List, retrieve, and delete documents
- **System Statistics**: Monitor storage and system health

## Architecture

### System Design

```
┌─────────────────┐
│   Frontend UI   │  
└────────┬────────┘
         │ HTTP REST API
         ▼
┌─────────────────────────────────────────────┐
│        FastAPI Application Server           │
├─────────────────────────────────────────────┤
│ Routes Layer                                │
│  ├── Upload Route    (POST /api/upload/*)   │
│  ├── Query Route     (POST /api/query/*)    │
│  └── Manage Route    (GET/DELETE /api/*)    │
└────────┬────────────────────┬───────────────┘
         │                    │
    ┌────▼────────┬───────────▼─────┐
    │             │                 │
    ▼             ▼                 ▼
┌──────────┐  ┌────────────┐  ┌──────────────┐
│Ingestion │  │Processing  │  │  Retrieval   │
│ Service  │  │  Service   │  │   Service    │
└──────────┘  └────────────┘  └──────────────┘
    │             │                 │
    │             ▼                 │
    │    ┌─────────────────┐        │
    │    │   Text Parser   │        │
    │    │   Tokenizer     │        │
    │    │   Chunker       │        │
    │    └─────────────────┘        │
    │                               │
    │        ┌──────────────────────┤
    │        │                      │
    │        ▼                      ▼
    │    ┌─────────────┐    ┌──────────────────┐
    │    │ TF-IDF      │    │ Keyword Matching │
    │    │ Vectorizer  │    │ Similarity Calc  │
    │    └─────────────┘    └──────────────────┘
    │
    └──────────────────────┬──────────────────┐
                          │                  │
                   ┌──────▼──────┐    ┌─────▼──────┐
                   │Answer Gen   │    │ Storage    │
                   │Service      │    │ Manager    │
                   └─────────────┘    └────────────┘
                          │                  │
                          └────────┬─────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  JSON Storage    │
                        │ (Persistence)    │
                        └──────────────────┘
```

### Components

#### 1. **Ingestion Service** (`app/services/ingestion.py`)
- Validates and ingests text and JSON documents
- Handles content cleaning and normalization
- Converts JSON structures to readable text
- Orchestrates document storage

#### 2. **Text Processing Service** (`app/services/processing.py`)
- **Chunking**: Splits text into overlapping chunks (default: 500 chars with 50-char overlap)
- **Preprocessing**: Normalizes whitespace, removes control characters
- **Tokenization**: Converts text to tokens for analysis
- **Keyword Extraction**: Identifies top keywords (filters stop words)
- **Statistics**: Calculates text metrics (word count, unique words, etc.)

#### 3. **Retrieval Service** (`app/services/retrieval.py`)
- **TF-IDF Method**: Uses scikit-learn's TfidfVectorizer for semantic similarity
  - Builds sparse matrices for efficient computation
  - Supports bigrams for context
  - Uses cosine similarity for ranking
- **Keyword Matching**: Implements Jaccard similarity with term frequency weighting
  - Combines set-based overlap with term frequency
  - Lightweight alternative to TF-IDF
- **Index Management**: Builds and refreshes indices when documents are added/removed

#### 4. **Answer Generation Service** (`app/services/answer_generation.py`)
- Extracts relevant sentences from retrieved chunks
- Constructs coherent answers by combining multiple sentences
- Calculates confidence scores based on:
  - Average relevance of retrieved chunks
  - Number of chunks found (more is better up to a point)
- Creates source references with document titles and chunk IDs
- Validates answer relevance to question

#### 5. **Storage Manager** (`app/storage.py`)
- **In-Memory Storage**: Fast access for queries
- **JSON Persistence**: Saves to disk for durability
- **Thread Safety**: Uses locks for concurrent access
- **Data Structure**:
  - Documents: `{doc_id: Document}`
  - Chunks Index: `{chunk_id: Chunk}` for quick access
  - Automatic saving on modifications

#### 6. **API Routes**
- **Upload** (`app/routes/upload.py`): POST endpoints for text, JSON, and files
- **Query** (`app/routes/query.py`): POST endpoint for answering questions
- **Manage** (`app/routes/manage.py`): GET/DELETE endpoints for document management

## API Endpoints

### Upload Endpoints

**POST `/api/upload/text`** - Upload text content
```json
{
  "title": "Document Title",
  "content": "Full text content...",
  "doc_type": "text"
}
```

**POST `/api/upload/json`** - Upload JSON content
```json
{
  "title": "JSON Document",
  "json_content": "{\"key\": \"value\"}"
}
```

**POST `/api/upload/file/text`** - Upload text file (.txt, .md, .csv)
```
Content-Type: multipart/form-data
- title: string
- file: file
```

### Query Endpoint

**POST `/api/query/answer`** - Ask a question
```json
{
  "question": "What is the main topic?",
  "top_k": 3,
  "retrieval_method": "tfidf"
}
```

Response:
```json
{
  "question": "What is the main topic?",
  "answer": "The main topic is...",
  "retrieved_chunks": [
    {
      "chunk_id": "uuid",
      "doc_id": "uuid",
      "content": "Chunk text...",
      "relevance_score": 0.85,
      "source_title": "Document Title"
    }
  ],
  "confidence": 0.78
}
```

### Management Endpoints

- `GET /api/manage/documents` - List all documents
- `GET /api/manage/documents/{doc_id}` - Get specific document
- `DELETE /api/manage/documents/{doc_id}` - Delete a document
- `GET /api/manage/documents/{doc_id}/stats` - Document statistics
- `GET /api/manage/stats` - System statistics
- `POST /api/manage/refresh-index` - Refresh TF-IDF index

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip or virtual environment manager

### Installation

1. **Navigate to project directory**
```bash
cd /home/angga/Project/AION_TEST
```

2. **Activate virtual environment** (already created)
```bash
source bin/activate
```

3. **Install dependencies** (already done, but you can reinstall)
```bash
pip install fastapi uvicorn scikit-learn numpy
```

4. **Run the server**
```bash
cd /home/angga/Project/AION_TEST
python -m app.main
```

The server will start at `http://localhost:8000`

### First Steps

1. Open interactive API documentation: `http://localhost:8000/docs`
2. Upload a document using POST `/api/upload/text`
3. Ask a question using POST `/api/query/answer`
4. View uploaded documents: `GET /api/manage/documents`

## Design Trade-offs and Decisions

### 1. **Retrieval Method Selection**

**TF-IDF vs Keyword Matching:**
- **TF-IDF** (Default):
  - ✅ Better for large documents with diverse vocabulary
  - ✅ Considers term frequency and document frequency
  - ❌ Higher computational cost
  - ❌ Requires vectorization step
  
- **Keyword Matching**:
  - ✅ Lighter weight and faster
  - ✅ No vectorization overhead
  - ❌ Less accurate for complex queries
  - ❌ Struggles with synonyms

**Decision**: Support both methods, default to TF-IDF for better accuracy.

### 2. **Chunking Strategy**

**Overlapping Chunks (current):**
- Default: 500 characters with 50-character overlap
- ✅ Prevents important information at chunk boundaries from being lost
- ✅ Provides context for boundary cases
- ❌ Slight data redundancy
- ❌ More storage overhead

**Alternative**: Fixed-size non-overlapping chunks would be simpler but risk losing context.

**Decision**: Use overlapping chunks for better coverage.

### 3. **Answer Generation Approach**

**Sentence-based Extraction:**
- Extract relevant sentences from retrieved chunks
- ✅ Ensures answers are grounded in source material
- ✅ Simple and interpretable
- ❌ Doesn't synthesize across documents
- ❌ Answer quality depends on source formatting

**Alternative**: Could train a seq2seq model, but violates "no external AI APIs" requirement.

**Decision**: Stick to sentence extraction with composition for simplicity and transparency.

### 4. **Storage Strategy**

**In-Memory with JSON Persistence:**
- ✅ Fast query performance
- ✅ Simple to implement
- ✅ Easy to inspect (JSON format)
- ❌ Not suitable for very large-scale systems
- ❌ Synchronous storage (could slow down writes)

**Alternatives**: Database (SQL/NoSQL), distributed cache (Redis).

**Decision**: In-memory + JSON for simplicity, suitable for development/demo scale.

### 5. **Confidence Scoring**

**Formula**: `(avg_relevance_score × 0.7) + (chunk_count_normalized × 0.3)`

- More weight on actual relevance scores
- Chunk count as secondary factor (more sources = more confidence)
- ✅ Interpretable and explainable
- ❌ Simple heuristic, not learned from data

**Decision**: Use weighted combination for transparency.

### 6. **Text Preprocessing**

**Current Approach**:
- Normalize whitespace
- Remove control characters
- Keep punctuation
- Minimal stop word removal (only in TF-IDF)

**Trade-offs**:
- ✅ Preserves document structure (punctuation, formatting)
- ✅ Avoids over-preprocessing
- ❌ Less aggressive than some NLP pipelines

**Decision**: Minimal preprocessing to preserve original content.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Document Upload | O(n) | n = document length, includes chunking |
| Text Chunking | O(n) | Linear scan with fixed chunk size |
| TF-IDF Indexing | O(d × t) | d = documents, t = terms |
| TF-IDF Query | O(log t) | After vectorization |
| Keyword Query | O(d × c) | d = documents, c = chunks |
| Answer Generation | O(k) | k = retrieved chunks |

## Possible Enhancements

1. **Ranking Improvements**:
   - Add BM25 ranking
   - Implement learning-to-rank with click data
   - Use domain-specific term weighting

2. **Caching**:
   - Cache TF-IDF queries
   - Cache frequent question patterns
   - LRU cache for popular documents

3. **Filtering**:
   - Date-based filtering
   - Category/tag filtering
   - Document type filtering

4. **Advanced Processing**:
   - Named Entity Recognition (NER)
   - Semantic similarity (transformers with ONNX)
   - Query expansion and reformulation

5. **Scalability**:
   - Move to proper database
   - Implement caching layer
   - Distributed indexing
   - Async document processing

## Testing

Example workflow:

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/upload/text" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Basics",
    "content": "Python is a high-level programming language. It is easy to learn..."
  }'

# Ask a question
curl -X POST "http://localhost:8000/api/query/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Python?",
    "top_k": 3,
    "retrieval_method": "tfidf"
  }'

# List documents
curl "http://localhost:8000/api/manage/documents"
```

## File Structure

```
/AION_TEST/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── schemas.py               # Pydantic models
│   ├── storage.py               # Storage management
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py         # Data ingestion
│   │   ├── processing.py        # Text processing
│   │   ├── retrieval.py         # Content retrieval
│   │   └── answer_generation.py # Answer construction
│   └── routes/
│       ├── __init__.py
│       ├── upload.py            # Upload endpoints
│       ├── query.py             # Query endpoints
│       └── manage.py            # Management endpoints
├── qa_storage.json              # Persisted storage
├── README.md                    # This file
└── requirements.txt             # Python dependencies

venv/                             # Virtual environment
├── bin/
├── lib/
└── ...
```

## Conclusion

This QA service demonstrates core engineering skills in:
- **Architecture Design**: Modular, scalable design with clear separation of concerns
- **Data Processing**: Intelligent chunking and preprocessing strategies
- **Information Retrieval**: Dual retrieval methods with configurable ranking
- **System Design**: Thread-safe storage with persistence
- **API Design**: RESTful endpoints with proper error handling
- **Performance**: Trade-offs between accuracy and efficiency

The system is production-ready for small to medium-scale deployments and can be extended with additional features as needed.
