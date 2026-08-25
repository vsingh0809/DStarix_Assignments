import os
from dotenv import load_dotenv

load_dotenv()

class Settings:

    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Model Configurations
    LLM_MODEL: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.1  # Low temperature for factual, grounded research

    @classmethod
    def validate(cls) -> None:
        """Validates critical runtime configurations."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set in environment variables. "
                "Please configure your .env file."
            )

settings = Settings()