import os
from dotenv import load_dotenv

load_dotenv()

MODEL_KEY = os.environ.get("MODEL_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME")
PINECONE_KEY=os.environ.get("PINECONE_KEY")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
VECTOR_STORE_INDEX_NAME=os.environ.get("VECTOR_STORE_INDEX_NAME")
CHUNK_SIZE=int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP=int(os.environ.get("CHUNK_OVERLAP"))