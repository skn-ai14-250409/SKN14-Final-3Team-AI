from typing import List, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from fastapi import UploadFile

from src.router import Router
from src.slm.slm import SLM
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader
from src.config import VECTOR_STORE_INDEX_NAME

GENERAL_FAQ_CATEGORY = "general_banking_FAQs"
COMPANY_PRODUCTS_CATEGORY = "company_products"
COMPANY_RULES_CATEGORY = "company_rules"
INDUSTRY_POLICY_CATEGORY = "industry_policies_and_regulations"

class Orchestrator:
    def __init__(self):
        self.router = Router()
        self.slm = SLM()
        self.vector_store = VectorStore(VECTOR_STORE_INDEX_NAME)
    
    def run_workflow(self, prompt):
        route_category: str = self.router.route_prompt(prompt)
        print(f"Category: {route_category}")
        if route_category == GENERAL_FAQ_CATEGORY:
            response: str = self.slm.invoke(prompt)
            return response
        elif (  route_category == COMPANY_PRODUCTS_CATEGORY or 
                route_category == COMPANY_RULES_CATEGORY or
                route_category == INDUSTRY_POLICY_CATEGORY
            ):
            response: str = self.query_rag(prompt)
            return response
        raise Exception("Router didn't choose a valid category.")

    async def upload_docs_to_rag(self, file: UploadFile) -> List[str]:
        self.vector_store.get_index_ready()
        document_loader = DocumentLoader()
        documents: List[Document] = await document_loader.get_document_chunks(file)
        uploaded_ids: List[str] = self.vector_store.add_documents_to_index(documents)
        return uploaded_ids

    def query_rag(self, query: str) -> str:
        self.vector_store.get_index_ready()
        retrieved_content: List[str] = self.vector_store.similarity_search(query)
        prompt: List[SystemMessage, HumanMessage] = []
        prompt.append(
            SystemMessage(
                "Use the following retrieved contents to inform your response: " + str(retrieved_content)
            )
        )
        prompt.append(
            HumanMessage(query)
        )
        response: str = self.slm.invoke(prompt)
        return response
    
    def delete_all_vectors(self) -> None:
        if not self.vector_store.check_index_exists():
            raise Exception("Vectors cannot be deleted as index doesn't exist.")
        return self.vector_store.delete_all_vectors()