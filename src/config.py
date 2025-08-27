import os
from dotenv import load_dotenv

load_dotenv()

MODEL_KEY = os.environ.get("MODEL_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
VECTOR_DB_DIR = os.environ.get("VECTOR_DB_DIR")
CHUNK_SIZE=int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP=int(os.environ.get("CHUNK_OVERLAP"))