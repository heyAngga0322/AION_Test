# Fullstack QA Engine

A bespoke Question Answering (QA) service that enables you to ingest local documents and natively perform semantic retrieval, answering questions completely offline through raw algorithmic extraction.

## 1. Application Specification
- **System Type:** Extractive Natural Language Processing (NLP) QA Web Application.
- **Data Ingestion:** Upload multiple raw text documents `.txt`, Portable Document Formats `.pdf`, and JSON files `.json`.
- **Knowledge Representation:** The system chunks paragraphs and ranks relevant passages with BM25 probabilistic ranking within a lightweight SQLite mapping.
- **Response Format:** It returns exact, pristine paragraph chunks that best answer the provided query natively, avoiding any generative LLM behavior.

## 2. Technology Stack & Libraries
This repository is split deeply across a modern frontend and a mathematical backend layer.

**Frontend Layer:**
- **Angular (v18+)**: Configured specifically in **Zoneless Mode** (`provideZonelessChangeDetection`) removing `zone.js` for ultra-optimized bundle sizes and execution speed.
- **Angular Signals:** Utilized as the strict reactive primitive. 
- **Tailwind CSS**: Leveraging utility classes and glassmorphism UI styles for a premium user perspective.

**Backend Engine:**
- **FastAPI / Uvicorn**: A highly concurrent async API layer. 
- **rank-bm25**: Probabilistic ranking algorithm for information retrieval, replacing TF-IDF for better document length normalization and term saturation handling.
- **SQLModel / SQLAlchemy**: Pydantic-first Object Relational Mapping storing source documents and chunks gracefully via SQLite.
- **pypdf**: An internal parser safely extracting raw string constructs from binary PDF pages.

## 3. Hot Take: The Engineering Decision 🌶️
**The Decision**: We actively rejected modern GenAI LLMs (OpenAI, Anthropic) and standard Vector Database juggernauts (Pinecone, Milvus, Qdrant) in favor of Localized BM25 Extractive QA mapped natively on SQLite arrays.

**The "Hot Take" Rationale**: The modern engineering landscape is suffering from a pervasive GenAI hype crisis—blindly throwing neural networks at fundamental text search problems. By constructing an Extractive QA system utilizing raw BM25 probabilistic ranking:
1. **Mathematical Verifiability**: Our system cannot "hallucinate". It either extracts the exact matching sentence in the localized text, or it rejects the query. 
2. **Zero Operating Cost**: We bypass millions of token deductions and prevent cloud vendor infrastructure lock-in completely.
3. **Substantially Lighter**: High-tier Vector DB orchestration requires vast memory footprints and cluster deployments. Fusing SQLite with standard Python internal memory bindings creates an instantly deployable runtime that calculates BM25 relevance scores at a fraction of the cost, requiring zero container configuration maintenance.

## 4. How to Run It
This application is designed identically to run containerized.

### Utilizing Docker Orchestration (Recommended)
From the root directory, simply deploy via:
```bash
docker-compose up --build -d
```
- Frontend interactive UI exposed at: `http://localhost:4200`
- Backend OpenAPI definitions exposed at: `http://localhost:8000/docs`

### Manual Development Runtime
If you are developing locally without Docker orchestration:

**Terminal 1: Backend API Target**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Reset Database (Start Over)
To wipe all ingested documents/chunks and recreate an empty database:

```bash
cd backend
source venv/bin/activate
python reset_db.py
```

By default, runtime DB data is stored at `app_data/database.db` in the repository root.

**Terminal 2: Frontend Client**
```bash
cd frontend/chat-ui
# Note: Node.js version MUST be >= v20.19
nvm use 22 # Use active environment
npm install
ng serve
```

## 5. How to Test It

### Backend Automated / Endpoint Testing
Since the environment utilizes FastAPI, a live Swagger integration testing environment is constructed immediately. Navigate to `http://localhost:8000/docs` while the backend is running to simulate native document ingestion and query behaviors from an un-opinionated browser client.

**Automated cURL Health Evaluation:**
```bash
# Evaluate QA engine networking bindings
curl -X POST http://localhost:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What architectural pattern is utilized within this codebase?"}'
```

### Frontend Codebase Specifications
The Angular frontend constructs test assertions across CLI native bindings:
```bash
cd frontend/chat-ui
npm run test
```
This executes the Karma runtime environment, allowing you to validate individual reactive signals without needing proxy binding active.

## 6. Using the Application Guide

### Step 1: Start the Application

**Docker (Recommended):**
```bash
docker-compose up --build -d
```
Then open `http://localhost:4200`

**Manual:**
```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend/chat-ui
ng serve
```

### Step 2: Upload Documents

In the **Knowledge Base** panel (left side):

1. **Enter a title** for your document (e.g., "Employee Handbook")
2. **Choose input method:**
   - **Upload file**: Click "Choose File" and select `.txt`, `.pdf`, or `.json`
   - **Paste text**: Type or paste text directly into the textarea
3. Click **"Upload & Process"**

The system will extract text, chunk into paragraphs, and build the BM25 index.

### Step 3: Ask Questions

In the **Ask the Context** panel (right side):

1. Type your question in the input field (e.g., "What is machine learning?")
2. Press **Enter** or click the send button
3. The system returns:
   - **Answer**: Relevant text chunks from your documents
   - **Sources**: Document titles the answer came from

### Example Workflow

```
1. Upload "ml_book.pdf" with title "ML Book"
2. Upload "company_policy.txt" with title "Company Policy"
3. Ask: "What is supervised learning?"
4. System returns matching paragraphs with source attribution
```

### Tips for Best Results

- Upload documents with **substantial text content** (not just tables of contents)
- Questions work best when they match the **terminology in your documents**
- The system returns **exact paragraphs** from your documents, not generated summaries
- If results are poor, try rephrasing your question using words from the documents
