from typing import List, Any

from langchain_core.documents import Document

from src.router import Router
from src.slm.slm import SLM
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader

GENERAL_FAQ_CATEGORY = "general_banking_FAQs"
COMPANY_PRODUCTS_CATEGORY = "company_products"
COMPANY_RULES_CATEGORY = "company_rules"
INDUSTRY_POLICY_CATEGORY = "industry_policies_and_regulations"

class Orchestrator:
    def __init__(self):
        self.router = Router()
        self.slm = SLM()
        self.vector_store = VectorStore()
    
    def run_workflow(self, prompt):
        route_category: str = self.router.route_prompt(prompt)
        print(f"Category: {route_category}")

        if route_category == GENERAL_FAQ_CATEGORY:
            response: str = self.slm.invoke(prompt)
            return response
        elif route_category in [
            COMPANY_PRODUCTS_CATEGORY,
            COMPANY_RULES_CATEGORY,
            INDUSTRY_POLICY_CATEGORY
        ]:
            response: str = self.query_rag(prompt, route_category)
            return response
        raise Exception("Router didn't choose a valid category.")
    
    def query_rag(self, query: str, category: str) -> str:
        results: List[Document] = self.vector_store.vector_store.similarity_search(
            query,
            k=3,
            filter={"category": category}
        )
        retrieved_content: List[str] = [doc.page_content for doc in results]
        prompt: List[Any] = []
        prompt.append(
            "Use the following retrieved contents to inform your response: " + str(retrieved_content)
        )
        prompt.append(query)
        response: str = self.slm.invoke(prompt)
        return response
    
    def upload_docs_to_rag(self, path: str, category: str) -> List[str]:
        document_loader = DocumentLoader()
        documents: List[Document] = document_loader.get_document_chunks(path, category)
        print(f"Chunks: {documents}")
        uploaded_ids: List[str] = self.vector_store.add_documents_to_index(documents)
        return uploaded_ids