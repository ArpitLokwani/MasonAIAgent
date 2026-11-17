
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Project paths
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
FAQ_FILE = KNOWLEDGE_BASE_PATH / "masonbro_faq.txt"
