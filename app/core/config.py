from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path 
BASE_DIR=Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"
    groq_api_key: str = ""
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "openai/gpt-oss-120b"
    top_k: int = 4
    max_retries: int = 1
    admin_api_key: str = "change-me"  # when we want to ad the new documents in kb that time only admin will have the right to chnage it so admin wil give the password and then chage it 
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")



#lru stands for least recently used and it stores/caches the result of the function so that when we call the function with the same argumrnt again it diesnt calculates it again but the stored one it returns back so that it becomes easy for expensive and recursive tasks 
@lru_cache
def get_settings() -> Settings:
    return Settings()