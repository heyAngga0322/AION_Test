# Fullstack QA Engine

A bespoke Question Answering (QA) service that enables you to ingest local documents and natively perform semantic retrieval, answering questions completely offline through raw algorithmic extraction.

## 1. Application Specification
- **System Type:** Extractive Natural Language Processing (NLP) QA Web Application.
- **Data Ingestion:** Upload multiple raw text documents `.txt` and Portable Document Formats `.pdf`.
- **Knowledge Representation:** The system chunks paragraphs, generating embeddings using localized Term Frequency-Inverse Document Frequency (TF-IDF) indexing algorithms within a lightweight SQLite mapping.
- **Response Format:** It returns exact, pristine paragraph chunks that best answer the provided query natively, avoiding any generative LLM behavior.

## 2. Technology Stack & Libraries
This repository is split deeply across a modern frontend and a mathematical backend layer.

**Frontend Layer:**
- **Angular (v18+)**: Configured specifically in **Zoneless Mode** (`provideZonelessChangeDetection`) removing `zone.js` for ultra-optimized bundle sizes and execution speed.
- **Angular Signals:** Utilized as the strict reactive primitive. 
- **Tailwind CSS**: Leveraging utility classes and glassmorphism UI styles for a premium user perspective.

**Backend Engine:**
- **FastAPI / Uvicorn**: A highly concurrent async API layer. 
- **scikit-learn**: Driving the heavy numerical vectorization (`TfidfVectorizer`) and mathematical ranking (Cosine Similarity).
- **SQLModel / SQLAlchemy**: Pydantic-first Object Relational Mapping storing chunk embeddings and tracking loaded sources gracefully via SQLite.
- **pypdf**: An internal parser safely extracting raw string constructs from binary PDF pages.

## 3. Hot Take: The Engineering Decision 🌶️
**The Decision**: We actively rejected modern GenAI LLMs (OpenAI, Anthropic) and standard Vector Database juggernauts (Pinecone, Milvus, Qdrant) in favor of Localized TF-IDF Extractive QA mapped natively on SQLite arrays.

**The "Hot Take" Rationale**: The modern engineering landscape is suffering from a pervasive GenAI hype crisis—blindly throwing neural networks at fundamental text search problems. By constructing an Extractive QA system utilizing raw TF-IDF mapping:
1. **Mathematical Verifiability**: Our system cannot "hallucinate". It either extracts the exact matching sentence in the localized text, or it rejects the query. 
2. **Zero Operating Cost**: We bypass millions of token deductions and prevent cloud vendor infrastructure lock-in completely.
3. **Substantially Lighter**: High-tier Vector DB orchestration requires vast memory footprints and cluster deployments. Fusing SQLite with standard Python internal memory bindings creates an instantly deployable runtime that calculates identical cosine similarities at a fraction of the cost, requiring zero container configuration maintenance.

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
