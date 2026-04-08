from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

class DocumentBase(SQLModel):
    title: str
    content: str

class Document(DocumentBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chunks: List["Chunk"] = Relationship(back_populates="document")

class ChunkBase(SQLModel):
    document_id: int = Field(foreign_key="document.id")
    text: str
    chunk_index: int

class Chunk(ChunkBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document: Optional[Document] = Relationship(back_populates="chunks")

# Pydantic models for API
class QueryRequest(SQLModel):
    question: str

class QueryResponse(SQLModel):
    answer: str
    sources: List[str]
