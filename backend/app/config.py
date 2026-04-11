from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM providers
    groq_api_key: str = ""
    ollama_base_url: str = "http://host.docker.internal:11434"  # optional offline fallback

    # HuggingFace (pyannote speaker diarization)
    hf_token: str = ""

    # MySQL
    mysql_host: str = "mysql"
    mysql_port: int = 3306
    mysql_database: str = "meeting_intelligence"
    mysql_user: str = "meeting_user"
    mysql_password: str = "meeting_password"

    # ChromaDB
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # Redis
    redis_url: str = "redis://redis:6379"

    @property
    def database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
