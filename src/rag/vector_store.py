from typing import List
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import PINECONE_KEY, MODEL_KEY, EMBEDDING_MODEL_NAME

class VectorStore:
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=PINECONE_KEY)
        self.index_name: str = index_name

    def create_index(self) -> None:
        # This uses the pinecone library
        index = self.pc.create_index(
            name=self.index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f'Created index: {index}')

    def check_index_exists(self) -> bool:
        # This uses the pinecone library
        if not self.pc.has_index(self.index_name):
            return False
        return True
    
    def get_index_ready(self) -> None:
        if not self.check_index_exists():
            self.create_index()
    
    def get_index(self) -> PineconeVectorStore:
        # This uses the Langchain-pinecone library
        embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            api_key=MODEL_KEY
        )
        index = self.pc.Index(self.index_name)
        vs_index = PineconeVectorStore(index=index, embedding=embedding_model)
        return vs_index

    def add_documents_to_index(self, documents: List[Document]) -> List[str]:
        vs_index: PineconeVectorStore = self.get_index()
        uuids = [str(uuid4()) for _ in range(len(documents))]
        list_of_ids: List[str] = vs_index.add_documents(documents=documents, ids=uuids)
        return list_of_ids
    
    def similarity_search(self, query: str) -> List[str]:
        vs_index: PineconeVectorStore = self.get_index()
        results: List[Document] = vs_index.similarity_search(
            query,
            k=3
        )
        results_contents: List[str] = [result.page_content for result in results]
        return results_contents
    
    def delete_all_vectors(self) -> None:
        vs_index: PineconeVectorStore = self.get_index()
        return vs_index.delete(delete_all=True)