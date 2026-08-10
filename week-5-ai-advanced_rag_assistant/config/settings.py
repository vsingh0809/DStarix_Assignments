import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    KNOWLEDGE_BASE_FILE: Path = DATA_DIR / "company_data.txt"
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configurations
    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    TEMPERATURE: float = 0

    # Text Splitter Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retriever Weights
    BM25_WEIGHT: float = 0.4
    VECTOR_WEIGHT: float = 0.6
    TOP_K_RESULTS: int = 3

    @classmethod
    def validate(cls) -> None:
        """Validates critical runtime configurations."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set in environment variables. "
                "Please configure your .env file."
            )

settings = Settings()