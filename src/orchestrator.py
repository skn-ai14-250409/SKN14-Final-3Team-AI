from typing import List, Any, Sequence, Union, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from fastapi import UploadFile
from pydantic import BaseModel, Field

from src.slm.slm import SLM
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader, process_folder_and_get_chunks
from src.config import VECTOR_STORE_INDEX_NAME


# 카테고리 상수 정의
GENERAL_FAQ_CATEGORY = "general_banking_FAQs"
COMPANY_PRODUCTS_CATEGORY = "company_products"
COMPANY_RULES_CATEGORY = "company_rules"
INDUSTRY_POLICY_CATEGORY = "industry_policies_and_regulations"

NO_ANSWER_MSG = "해당 정보를 찾을 수 없습니다."

_SYSTEM_PROMPT_TEMPLATE = """당신은 KB금융그룹의 내부 문서를 기반으로 답변하는 AI 어시스턴트입니다.

지침:
1) 제공된 <검색된_문서>에 질문과 관련된 정보가 있다면, 그 정보를 바탕으로 정확한 답변을 제공하세요.
2) 검색된 문서에 질문과 관련된 내용이 전혀 없거나 매우 부족하다면, 정확히 "해당 정보를 찾을 수 없습니다"라고 답하세요.
3) 추측하거나 문서에 없는 내용을 만들어내지 마세요.
4) 답변은 한국어로 명확하고 구체적으로 작성하세요.
5) 가능한 경우 문서의 정확한 내용을 인용하여 답변하세요.

<검색된_문서>
{context}
</검색된_문서>
"""

# Router 클래스 정의
class RouterResponse(BaseModel):
    category: Literal[
        "general_banking_FAQs",
        "industry_policies_and_regulations",
        "company_rules",
        "company_products"
    ] = Field(
        ...,
        description="Classify the query into one of the following categories: \
        1. general_banking_faqs – for common, non-company-specific banking knowledge (e.g., what is a checking account); \
        2. industry_policies_and_regulations – for banking industry compliance or regulatory queries (e.g., KYC, AML, Basel III); \
        3. company_rules – for internal rules and HR policies of our bank (e.g., vacation policy, dress code); \
        4. company_products – for questions about specific products offered by our bank (e.g., loan products, account types, interest rates)."
    )

class Router:
    def __init__(self):
        self.slm = SLM()
        self.router_response_cls = RouterResponse

    def route_prompt(self, prompt) -> str:
        prompt_category: RouterResponse = self.slm.get_structured_output(
            prompt,
            self.router_response_cls
        )
        return prompt_category.category

class Orchestrator:
    def __init__(self):
        self.router = Router()
        self.slm = SLM()
        self.vector_store = VectorStore(VECTOR_STORE_INDEX_NAME)

    def run_workflow(self, prompt: str) -> str:
        """프롬프트를 분석하여 적절한 처리 방식을 결정하고 응답을 반환합니다."""
        route_category: str = self.router.route_prompt(prompt)
        print(f"Category: {route_category}")

        if route_category == GENERAL_FAQ_CATEGORY:
            return self.slm.invoke(prompt)

        elif route_category in (COMPANY_PRODUCTS_CATEGORY,
                                COMPANY_RULES_CATEGORY,
                                INDUSTRY_POLICY_CATEGORY):
            result = self.query_rag(prompt)
            return result["response"]

        raise Exception("Router didn't choose a valid category.")

    async def upload_docs_to_rag(self, file: UploadFile) -> List[str]:
        """업로드된 파일을 벡터 스토어에 추가합니다."""
        self.vector_store.get_index_ready()
        documents: List[Document] = await DocumentLoader().get_document_chunks(file)
        if not documents:
            return []
        uploaded_ids: List[str] = self.vector_store.add_documents_to_index(documents)
        return uploaded_ids

    def _normalize_retrieved(self, items: Sequence[Union[Document, str, Dict[str, Any]]]) -> List[Document]:
        """검색 결과를 Document 리스트로 정규화합니다."""
        norm: List[Document] = []
        for it in items or []:
            if isinstance(it, Document):
                norm.append(it)
            elif isinstance(it, str):
                norm.append(Document(page_content=it, metadata={"source_type": "raw_text"}))
            elif isinstance(it, dict):
                text = it.get("page_content") or it.get("text") or it.get("content") or ""
                meta = it.get("metadata") or {}
                norm.append(Document(page_content=text, metadata=meta))
            else:
                norm.append(Document(page_content=str(it), metadata={"source_type": "unknown"}))
        return norm

    def _format_context(self, docs: Sequence[Document]) -> str:
        """Document 리스트를 컨텍스트 문자열로 포맷합니다."""
        lines = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("relative_path") or meta.get("file_name") or meta.get("source_type") or "unknown_source"
            snippet = (d.page_content or "").strip()
            if not snippet:
                continue
            lines.append(f"[source: {src}]\n{snippet}")
        return "\n---\n".join(lines)

    def _build_system_message(self, retrieved: Sequence[Union[Document, str]]) -> SystemMessage:
        """검색된 문서들로부터 시스템 메시지를 생성합니다."""
        ctx = self._format_context(retrieved)
        return SystemMessage(_SYSTEM_PROMPT_TEMPLATE.format(context=ctx))

    def query_rag(self, query: str) -> Dict[str, Any]:
        """RAG를 사용하여 쿼리에 응답합니다."""
        self.vector_store.get_index_ready()
        raw_retrieved = self.vector_store.similarity_search(query)

        # 검색 결과 정규화
        retrieved_docs = self._normalize_retrieved(raw_retrieved)

        # 컨텍스트가 비었으면 바로 종료
        if not retrieved_docs:
            return {"response": NO_ANSWER_MSG, "sources": []}

        context_text = self._format_context(retrieved_docs)
        if not context_text.strip():
            return {"response": NO_ANSWER_MSG, "sources": []}
        
        # 검색된 문서의 최소 길이 체크 (너무 짧으면 관련성이 낮을 가능성)
        if len(context_text) < 50:
            return {"response": NO_ANSWER_MSG, "sources": []}

        # LLM 호출
        prompt = [self._build_system_message(retrieved_docs), HumanMessage(query)]
        try:
            response: str = self.slm.invoke(prompt)
        except Exception:
            response = None

        # 소스 메타데이터 추출
        sources = []
        for d in retrieved_docs:
            meta = dict(d.metadata or {})
            if "relative_path" not in meta:
                rf = meta.get("root_folder")
                fn = meta.get("file_name")
                if rf and fn:
                    meta["relative_path"] = f"{rf}/{fn}"
            sources.append(meta)

        return {
            "response": response or NO_ANSWER_MSG,
            "sources": sources
        }

    def upload_folder_to_rag(self, folder_path: str) -> List[str]:
        """폴더의 모든 문서를 벡터 스토어에 업로드합니다."""
        self.vector_store.get_index_ready()
        documents: List[Document] = process_folder_and_get_chunks(folder_path)
        if not documents:
            return []
        uploaded_ids: List[str] = self.vector_store.add_documents_to_index(documents)
        return uploaded_ids

    def delete_all_vectors(self) -> None:
        """벡터 스토어의 모든 벡터를 삭제합니다."""
        if not self.vector_store.check_index_exists():
            raise Exception("Vectors cannot be deleted as index doesn't exist.")
        return self.vector_store.delete_all_vectors()
    
    def delete_vectors_by_condition(self, field: str, value: str) -> int:
        """특정 조건에 맞는 벡터만 삭제합니다."""
        if not self.vector_store.check_index_exists():
            raise Exception("Vectors cannot be deleted as index doesn't exist.")
        return self.vector_store.delete_vectors_by_condition(field, value)