import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TEMP_DIR: Path = BASE_DIR / "temp_uploads"

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    LLM_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    TEMPERATURE: float = 0

    # UI / RAG Defaults
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_RETRIEVAL_COUNT: int = 3

    @classmethod
    def validate(cls) -> None:
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from the environment variables.")
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()