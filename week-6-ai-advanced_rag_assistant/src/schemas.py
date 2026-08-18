from pydantic import BaseModel, Field

class RAGConfig(BaseModel):
    """Schema to hold dynamic RAG tuning parameters passed from the UI."""
    chunk_size: int = Field(default=800, description="Size of text chunks")
    chunk_overlap: int = Field(default=100, description="Overlap between text chunks")
    retrieval_count: int = Field(default=3, description="Number of documents to retrieve (k)")