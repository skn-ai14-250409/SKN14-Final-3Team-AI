from typing import List
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import EMBEDDING_MODEL_NAME, COLLECTION_NAME, VECTOR_DB_DIR, MODEL_KEY

class VectorStore:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key=MODEL_KEY)
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_model,
            persist_directory=VECTOR_DB_DIR
        )
    
    def add_documents_to_index(self, documents: List[Document]) -> List[str]:
        uuids = [str(uuid4()) for _ in range(len(documents))]
        list_of_ids: List[str] = self.vector_store.add_documents(documents=documents, ids=uuids)
        return list_of_ids
    
    def similarity_search(self, query: str) -> List[str]:
        results: List[Document] = self.vector_store.similarity_search(
            query,
            k=3
        )
        results_contents: List[str] = [result.page_content for result in results]
        return results_contents