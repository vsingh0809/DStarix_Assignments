from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger("DocumentLoader")

def ensure_sample_data(file_path: Path) -> None:
    """Creates a default company policy document if no custom document exists."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        logger.info(f"Knowledge base missing. Writing default sample file at: {file_path}")
        sample_text = (
            "AcmeCorp HR & Operations Policy:\n"
            "1. Remote Work Policy: Employees can work remotely up to 3 days per week.\n"
            "2. IT Support Desk: Open tickets online or reach internal hotline at 555-0199.\n"
            "3. Expense Submissions: Submit all business travel receipts via Concur within 14 days.\n"
            "4. Strategic Objectives: Q3 primary focus is legacy database migration to cloud platforms."
        )
        file_path.write_text(sample_text, encoding="utf-8")

def load_and_split_documents() -> List[Document]:
    """Loads documents from disk and splits them into configured chunk sizes."""
    ensure_sample_data(settings.KNOWLEDGE_BASE_FILE)
    
    logger.info(f"Loading document from {settings.KNOWLEDGE_BASE_FILE}...")
    loader = TextLoader(str(settings.KNOWLEDGE_BASE_FILE), encoding="utf-8")
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(raw_docs)
    logger.info(f"Successfully processed document into {len(chunks)} text chunks.")
    return chunks