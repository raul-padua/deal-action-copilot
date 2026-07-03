import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file) for local dev.
# On Vercel, env vars come from the dashboard and this is a no-op.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
QDRANT_URL = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or None

DATA_DIR = Path(__file__).resolve().parent / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
COLLECTION_NAME = "socure_knowledge"

# Bound on the research loop: max LLM turns before forced generation.
MAX_RESEARCH_TURNS = 3
