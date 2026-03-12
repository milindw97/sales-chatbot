"""
Configuration management for Sales Chatbot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
DB_DIR = DATA_DIR / "db"

# Ensure directories exist
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")

# Database Configuration
DB_PATH = PROJECT_ROOT / os.getenv("DB_PATH", "data/db/sales_calls.db")
FAISS_INDEX_PATH = PROJECT_ROOT / os.getenv("FAISS_INDEX_PATH", "data/db/faiss_index")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Conversation History
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))


# Validation
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set but LLM_PROVIDER=gemini")

if EMBEDDING_PROVIDER == "gemini" and not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set but EMBEDDING_PROVIDER=gemini")
