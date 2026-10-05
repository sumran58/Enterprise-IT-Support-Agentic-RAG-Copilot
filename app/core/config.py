from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path 
BASE_DIR=Path(__file__).resolve().parent[2]