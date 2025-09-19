"""
실험용 LangGraph RAG 워크플로우

기존 orchestrator.py와 동일한 기능을 그래프 기반으로 구현
기존 코드는 그대로 유지하고 새로운 접근 방식을 시험해보기 위한 파일
"""

from typing import Dict, List, Any, TypedDict, Annotated
import logging
import re
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.slm.slm import SLM
from src.rag.vector_store import VectorStore
from src.intent_router import IntentRouter
from src.constants import (
    NO_ANSWER_MSG,
    MAIN_LAW, MAIN_RULE, MAIN_PRODUCT,
    GENERAL_FAQ_CATEGORY,
    COMPANY_PRODUCTS_CATEGORY,
    COMPANY_RULES_CATEGORY,
    INDUSTRY_POLICY_CATEGORY,
)

logger = logging.getLogger(__name__)

# LangGraph State 정의
class RAGState(TypedDict):
    """RAG 워크플로우의 상태를 관리하는 클래스"""
    messages: Annotated[List[BaseMessage], add]
    query: str
    category: str
    product_name: str
    retrieved_docs: List[Document]
    context_text: str
    response: str
    sources: List[Dict[str, Any]]
    is_first_question: bool  # 첫 턴 전처리용 상태 필드
    initial_intent: str
    initial_topic_summary: str  # 세션 제목(요약) 저장용

# Pydantic 모델: 함수 내부 중첩 정의를 모듈 수준으로 이동
class ProductNameResponse(BaseModel):
    product_name: str = Field(
        ...,
        description="질문에서 언급된 KB금융그룹 상품명만 추출하세요. 상품명이 없으면 빈 문자열을 반환하세요."
    )

class LangGraphRAGWorkflow:
    """LangGraph 기반 실험용 RAG 워크플로우"""

    def __init__(self):
        self.slm = SLM()
        self.vector_store = VectorStore()
        self.router = IntentRouter()
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구축"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("first_turn_preprocess", self._first_turn_preprocess)  # ✅ 추가
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("handle_general_faq", self._handle_general_faq)
        workflow.add_node("extract_product_name", self._extract_product_name)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("filter_relevance", self._filter_relevance)
        workflow.add_node("generate_response", self._generate_response)

        # 엣지 추가 (조건부 라우팅)
        workflow.add_edge(START, "first_turn_preprocess")        # ✅ START → 첫 턴 전처리
        workflow.add_edge("first_turn_preprocess", "classify_intent")  # ✅ 전처리 후 인텐트 분류

        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_category,
            {
                "general_faq": "handle_general_faq",
                "rag_needed": "extract_product_name"
            }
        )

        workflow.add_edge("handle_general_faq", END)
        workflow.add_edge("extract_product_name", "search_documents")
        workflow.add_edge("search_documents", "filter_relevance")
        workflow.add_edge("filter_relevance", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def _first_turn_preprocess(self, state: RAGState) -> RAGState:
        """세션의 첫 질문에 대해서만 실행되는 전처리 노드(최소 예외 처리).

        동작:
        - 세션 첫 질문인지 확인 (is_first_question)
        - router로 intent 획득(initial_intent) — 오류는 상위로 전달
        - SLM으로 챗봇 세션용 짧은 제목 생성(initial_topic_summary)
          (SLM 호출 실패에 대해서만 최소 폴백 처리)
        - is_first_question을 False로 설정하고 상태 반환
        """
        # 이미 처리되었다면 그대로 반환 (세션 내 1회 실행)
        if not state.get("is_first_question", True):
            return state

        query = state.get("query", "") or ""
        # router 호출에 대한 try/except는 제거 — 오류는 상위로 전달되어야 디버깅에 유리
        initial_intent = (self.router.route_prompt(query) or "").strip()

        # 제목 생성: LLM 호출만 한 번의 예외 처리 (운영상 폴백 허용)
        system_message = SystemMessage(
            "당신은 '챗봇 세션 제목 생성기'입니다. 사용자의 질문을 보고 "
            "이 세션을 대표할 매우 간결한 한 줄 제목을 생성하세요."
        )
        human_prompt = (
            f"사용자 질문: {query}\n\n"
            "출력: 제목 텍스트만 한 줄로 출력\n"
            "- 한국어로 간결하게 (17 어절 권장)\n"
            "- 불필요한 설명 없이 제목만 반환\n"
        )
        messages = [system_message, HumanMessage(human_prompt)]

        session_title = ""
        MAX_TITLE_CHARS = 60

        try:
            raw = (self.slm.invoke(messages) or "").strip()
            if raw:
                session_title = raw.splitlines()[0].strip()
                if len(session_title) > MAX_TITLE_CHARS:
                    session_title = session_title[:MAX_TITLE_CHARS].rstrip() + "..."
        except Exception as e:
            # 최소 폴백: intent 또는 질문의 앞부분 사용
            logger.warning(f"[GRAPH] first_turn_preprocess: title generation failed: {e}")
            session_title = ""

        if not session_title:
            # LLM 실패 또는 빈 결과일 때 폴백: intent 우선, 없으면 질문 앞부분
            session_title = initial_intent or (query.strip().splitlines()[0][:MAX_TITLE_CHARS] or "새로운 질문")

        logger.info(
            f"[GRAPH] First-turn preprocess done. intent={initial_intent!r}, session_title={session_title!r}"
        )

        return {
            **state,
            "is_first_question": False,
            "initial_intent": initial_intent,
            # 기존 필드명(initial_topic_summary)을 세션 제목 저장 용도로 재사용
            "initial_topic_summary": session_title,
        }

    def _classify_intent(self, state: RAGState) -> RAGState:
        """인텐트 분류 노드"""
        query = state["query"]
        category = self.router.route_prompt(query)

        logger.info(f"[GRAPH] Classified category: {category}")

        return {
            **state,
            "category": category
        }

    def _route_by_category(self, state: RAGState) -> str:
        """카테고리에 따른 라우팅 결정"""
        category = state["category"]

        if category == GENERAL_FAQ_CATEGORY:
            return "general_faq"
        elif category in [COMPANY_PRODUCTS_CATEGORY, COMPANY_RULES_CATEGORY, INDUSTRY_POLICY_CATEGORY]:
            return "rag_needed"
        else:
            return "rag_needed"  # 기본적으로 RAG 사용

    def _handle_general_faq(self, state: RAGState) -> RAGState:
        """일반 FAQ 처리 노드"""
        query = state["query"]

        # 일반 FAQ용 시스템 메시지
        system_message = SystemMessage("""당신은 KB금융그룹의 고객 상담 전문가입니다.

지침:
1) 일반적인 금융 상식이나 KB금융그룹의 기본 정보에 대해 친근하고 정확하게 답변하세요.
2) 복잡한 금융 용어는 쉽게 풀어서 설명하세요.
3) 구체적인 상품 정보나 규정이 필요한 경우, "상세한 상담을 위해 KB금융그룹에 직접 문의하시기 바랍니다"라고 안내하세요.
4) 고객의 상황에 맞는 일반적인 조언을 제공하되, 개인 맞춤 상담은 별도 안내하세요.
5) 항상 정중하고 도움이 되는 어조로 답변하세요.
6) 답변은 5줄 이내로 간결하게 작성하세요.""")
        messages = [system_message, HumanMessage(query)]
        response = self.slm.invoke(messages)

        logger.info(f"[GRAPH] General FAQ response generated")

        return {
            **state,
            "messages": state.get("messages", []) + messages,
            "response": response,
            "sources": []
        }

    def _extract_product_name(self, state: RAGState) -> RAGState:
        """상품명 추출 노드"""
        query = state["query"]
        product_name = self._extract_product_name_from_question(query)

        logger.info(f"[GRAPH] Extracted product name: '{product_name}'")

        return {
            **state,
            "product_name": product_name
        }

    def _search_documents(self, state: RAGState) -> RAGState:
        """문서 검색 노드"""
        query = state["query"]
        product_name = state.get("product_name", "")
        category = state["category"]

        self.vector_store.get_index_ready()
        raw_retrieved = []

        # 1차: 질문 키워드로 정확한 파일명 매칭 (최우선)
        try:
            query_keywords = self._extract_keywords_from_query(query)
            exact_filename = self._find_exact_filename_match(query_keywords)

            if exact_filename:
                raw_retrieved = self.vector_store.similarity_search_by_filename(query, exact_filename)
                logger.info(f"[GRAPH] Exact filename match: {exact_filename}")
        except Exception as e:
            logger.warning(f"[GRAPH] Filename matching failed: {e}")
            # 파일명 매칭 실패 시 계속 진행

        # 2차: 상품명 기반 검색 (정확한 파일명 매칭이 실패한 경우만)
        if not raw_retrieved and product_name:
            try:
                # 1차: 파일명 정확 매칭
                filename_with_underscores = product_name.replace(" ", "_") + ".pdf"
                raw_retrieved = self.vector_store.similarity_search_by_filename(query, filename_with_underscores)

                if not raw_retrieved:
                    # 2차: 키워드 검색
                    keywords = self._extract_keywords_from_product_name(product_name)
                    raw_retrieved = self.vector_store.similarity_search_by_keywords(query, keywords)

                    if not raw_retrieved:
                        # 3차: 일반 검색
                        raw_retrieved = self.vector_store.similarity_search(query)
            except Exception as e:
                logger.warning(f"[GRAPH] Product name search failed: {e}")
                raw_retrieved = self.vector_store.similarity_search(query)
        else:
            # 3차: 카테고리별 검색 또는 일반 검색
            try:
                if category == COMPANY_PRODUCTS_CATEGORY:
                    raw_retrieved = self.vector_store.similarity_search_by_folder(query, MAIN_PRODUCT)
                elif category == COMPANY_RULES_CATEGORY:
                    raw_retrieved = self.vector_store.similarity_search_by_folder(query, MAIN_RULE)
                elif category == INDUSTRY_POLICY_CATEGORY:
                    raw_retrieved = self.vector_store.similarity_search_by_folder(query, MAIN_LAW)
                else:
                    raw_retrieved = self.vector_store.similarity_search(query)
            except Exception as e:
                logger.warning(f"[GRAPH] Category search failed: {e}")
                raw_retrieved = self.vector_store.similarity_search(query)

        # 문서 정규화
        retrieved_docs = self._normalize_retrieved(raw_retrieved)

        logger.info(f"[GRAPH] Retrieved {len(retrieved_docs)} documents")

        return {
            **state,
            "retrieved_docs": retrieved_docs
        }

    def _find_exact_filename_match(self, query_keywords: List[str]) -> str:
        """질문 키워드로 정확한 파일명 찾기 (동적 방식)"""
        try:
            # 파일명에서 자동으로 키워드 추출
            available_files = self.vector_store.get_available_files()

            if not available_files:
                logger.warning("[GRAPH] No available files found")
                return ""

            best_match = ""
            max_score = 0.0
            min_threshold = 0.3  # 최소 임계값 설정

            for filename in available_files:
                try:
                    # 파일명에서 키워드 자동 추출
                    file_keywords = self._extract_keywords_from_filename(filename)

                    # 질문 키워드와 매칭 점수 계산
                    score = self._calculate_keyword_match_score(query_keywords, file_keywords)

                    if score > max_score and score >= min_threshold:
                        max_score = score
                        best_match = filename

                except Exception as e:
                    logger.error(f"[GRAPH] Error processing filename {filename}: {e}")
                    continue

            logger.info(f"[GRAPH] Best filename match: {best_match} (score: {max_score:.2f})")
            return best_match if max_score >= min_threshold else ""

        except Exception as e:
            logger.error(f"[GRAPH] Error in filename matching: {e}")
            return ""

    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """질문에서 핵심 키워드 추출"""
        try:
            keywords = []
            words = query.split()

            # 불용어 제거
            stop_words = ["은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "는", "도", "만", "부터", "까지", "에서", "에게", "한테", "와", "과", "의", "가", "이", "을", "를", "에", "에서", "로", "으로", "부터", "까지", "만", "도", "는", "은", "이", "가", "을", "를", "에", "의", "로", "으로", "뭐", "있어", "있나", "알려", "주세요", "안내", "정보"]

            for word in words:
                if len(word) > 1 and word not in stop_words:
                    keywords.append(word)

            return keywords
        except Exception as e:
            logger.error(f"[GRAPH] Error extracting keywords from query: {e}")
            return []

    def _extract_keywords_from_filename(self, filename: str) -> List[str]:
        """파일명에서 키워드 자동 추출"""
        try:
            # 파일명 정리 (확장자 제거, 언더스코어/공백 처리)
            clean_name = filename.replace(".pdf", "").replace("KB_", "").replace("_", " ")

            # 한글, 영문, 숫자만 추출
            keywords = re.findall(r'[가-힣a-zA-Z0-9]+', clean_name)

            # 길이가 1인 단어 제거
            keywords = [kw for kw in keywords if len(kw) > 1]

            return keywords
        except Exception as e:
            logger.error(f"[GRAPH] Error extracting keywords from filename: {e}")
            return []

    def _calculate_keyword_match_score(self, query_keywords: List[str], file_keywords: List[str]) -> float:
        """키워드 매칭 점수 계산 (개선된 버전)"""
        try:
            if not query_keywords or not file_keywords:
                return 0.0

            # 정확한 매칭 (가장 높은 점수)
            exact_matches = len(set(query_keywords) & set(file_keywords))

            # 부분 매칭 (포함 관계) - 중복 계산 방지
            partial_matches = 0
            matched_query_words = set()

            for q_kw in query_keywords:
                for f_kw in file_keywords:
                    if q_kw in f_kw or f_kw in q_kw:
                        if q_kw not in matched_query_words:  # 중복 방지
                            partial_matches += 0.3
                            matched_query_words.add(q_kw)

            # 가중치 적용
            total_score = exact_matches * 1.0 + partial_matches * 0.3

            # 정규화 (질문 키워드 수로 나누기)
            return total_score / len(query_keywords) if query_keywords else 0.0
        except Exception as e:
            logger.error(f"[GRAPH] Error calculating keyword match score: {e}")
            return 0.0

# ----------------------------------------------------------------------------------------------------------------------------

    def _filter_relevance(self, state: RAGState) -> RAGState:
        """관련성 필터링 노드"""
        docs = state["retrieved_docs"]
        query = state["query"]

        filtered_docs = self._filter_by_relevance_score(docs, query)
        context_text = self._format_context(filtered_docs)

        logger.info(f"[GRAPH] Filtered to {len(filtered_docs)} relevant documents")

        return {
            **state,
            "retrieved_docs": filtered_docs,
            "context_text": context_text
        }

    def _generate_response(self, state: RAGState) -> RAGState:
        """응답 생성 노드"""
        query = state["query"]
        context_text = state["context_text"]
        retrieved_docs = state["retrieved_docs"]
        category = state["category"]

        if not context_text.strip():
            return {
                **state,
                "response": NO_ANSWER_MSG,
                "sources": []
            }

        # 카테고리별 시스템 메시지 생성
        system_message = self._build_category_specific_system_message(category, context_text)
        messages = [system_message, HumanMessage(query)]

        response = self.slm.invoke(messages)

        # 소스 정보 추출 - 수정된 부분
        sources = []
        for doc in retrieved_docs:
            metadata = doc.metadata or {}
            # 실제 문서 내용도 포함
            source_info = dict(metadata)
            source_info['text'] = doc.page_content  # 실제 문서 내용 추가
            source_info['page_content'] = doc.page_content  # 호환성을 위해 추가
            sources.append(source_info)

        logger.info(f"[GRAPH] Generated response with {len(sources)} sources")

        return {
            **state,
            "messages": state.get("messages", []) + messages,
            "response": response,
            "sources": sources
        }

    def run_workflow(self, query: str) -> Dict[str, Any]:
        """워크플로우 실행"""
        initial_state = RAGState(
            messages=[],
            query=query,
            category="",
            product_name="",
            retrieved_docs=[],
            context_text="",
            response="",
            sources=[],
            # ✅ 첫 턴 전처리 관련 초기값
            is_first_question=True,
            initial_intent="",
            initial_topic_summary="",
        )

        try:
            final_state = self.workflow.invoke(initial_state)

            return {
                "response": final_state["response"],
                "sources": final_state["sources"],
                "category": final_state.get("category", "unknown"),
                "product_name": final_state.get("product_name", ""),
                # ✅ 참고용 반환
                "initial_intent": final_state.get("initial_intent", ""),
                "initial_topic_summary": final_state.get("initial_topic_summary", ""),
            }
        except Exception as e:
            logger.error(f"[GRAPH] Workflow execution failed: {e}")
            return {
                "response": "처리 중 오류가 발생했습니다.",
                "sources": [],
                "category": "error",
                "product_name": "",
                "initial_intent": "",
                "initial_topic_summary": "",
            }

    # 기존 orchestrator의 헬퍼 메서드들 복사
    # 프롬프트 좀 더 구체적으로 작성.
    def _extract_product_name_from_question(self, question: str) -> str:
        """질문에서 상품명을 추출하는 메서드 (기존 orchestrator와 동일)"""
        try:
            extraction_prompt = f"""
                다음 질문에서 질문자의 의도와 가장 관련성이 높은 KB금융그룹 상품명만 추출하세요.
                질문: {question}

                규칙:
                1) 질문에서 명시적으로 언급된 상품명만 추출
                2) 질문의 맥락과 관련 없는 상품명은 추출하지 않음
                3) 상품명이 명확하지 않으면 빈 문자열 반환
                4) 예시는 참고용일 뿐, 질문과 무관한 상품명 추출 금지
            """

            product_response = self.slm.get_structured_output(
                extraction_prompt,
                ProductNameResponse
            )
            return product_response.product_name.strip()
        except Exception as e:
            logger.error(f"[GRAPH] Failed to extract product name: {e}")
            return ""

    def _extract_keywords_from_product_name(self, product_name: str) -> List[str]:
        """상품명에서 핵심 키워드 추출 (기존 orchestrator와 동일)"""
        keywords = []
        clean_name = product_name.replace("KB", "").strip()
        words = re.findall(r'[가-힣]+', clean_name)

        for word in words:
            if len(word) > 1:
                keywords.append(word)

        keyword_mapping = {
            "동반성장협약": ["동반성장", "협약", "상생"],
            "상생대출": ["상생", "대출"],
            "닥터론": ["닥터론"],
            "스마트론": ["스마트론"],
            "햇살론": ["햇살론"]
        }

        for key, mapped_keywords in keyword_mapping.items():
            if key in product_name:
                keywords.extend(mapped_keywords)

        return list(set(keywords))

    def _normalize_retrieved(self, items) -> List[Document]:
        """검색 결과 정규화 (기존 orchestrator와 동일)"""
        norm = []
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

    def _format_context(self, docs) -> str:
        """컨텍스트 포맷 (기존 orchestrator와 동일)"""
        lines = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("relative_path") or meta.get("file_name") or meta.get("source_type") or "unknown_source"
            snippet = (d.page_content or "").strip()
            if not snippet:
                continue
            lines.append(f"[source: {src}]\n{snippet}")
        return "\n---\n".join(lines)

    def _filter_by_relevance_score(self, docs: List[Document], query: str) -> List[Document]:
        """관련성 필터링 (기존 orchestrator와 동일)"""
        if not docs:
            return docs

        query_words = set(query.lower().split())
        scored_docs = []

        for doc in docs:
            content = (doc.page_content or "").lower()
            metadata = doc.metadata or {}

            score = 1.0

            # 키워드 매칭 보너스
            content_words = set(content.split())
            keyword_overlap = len(query_words.intersection(content_words))
            score += keyword_overlap * 0.1

            # 메타데이터 키워드 매칭 보너스
            metadata_keywords = metadata.get("keywords", [])
            for keyword in metadata_keywords:
                if keyword.lower() in query.lower():
                    score += 0.2

            # 파일명 매칭 보너스
            file_name = metadata.get("file_name", "").lower()
            for word in query_words:
                if word in file_name:
                    score += 0.3

            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        min_score = 1.0
        filtered_docs = [doc for score, doc in scored_docs if score >= min_score]

        return filtered_docs[:5]

    def _build_category_specific_system_message(self, category: str, context_text: str) -> SystemMessage:
        """카테고리별 시스템 메시지 (기존 orchestrator와 동일)"""
        if category == COMPANY_PRODUCTS_CATEGORY:
            system_prompt = """당신은 KB금융그룹의 금융상품 전문 상담사입니다.

지침:
1) 제공된 <검색된_문서>는 KB금융그룹의 공식 상품 정보입니다.
2) 상품의 특징, 조건, 금리, 한도, 신청방법 등을 정확하고 구체적으로 안내하세요.
3) 고객이 이해하기 쉽도록 친근하고 전문적인 어조로 답변하세요.
4) 상품 비교나 추천이 필요한 경우, 문서 내용을 바탕으로 객관적으로 설명하세요.
5) 문서에 없는 정보는 "추가 상담이 필요합니다"라고 안내하세요.
6) 가능한 경우 관련 상품이나 서비스도 함께 안내하세요.
7) 답변은 5줄 이내로 간결하게 작성하세요.

<검색된_문서>
{context}
</검색된_문서>"""

        elif category == COMPANY_RULES_CATEGORY:
            system_prompt = """당신은 KB금융그룹의 내부 규정 및 정책 전문가입니다.

지침:
1) 제공된 <검색된_문서>는 KB금융그룹의 공식 내부 규정과 정책입니다.
2) 규정의 목적, 적용 범위, 세부 조건, 절차 등을 정확하고 명확하게 설명하세요.
3) 복잡한 규정은 단계별로 나누어 이해하기 쉽게 설명하세요.
4) 관련 법령이나 상위 규정과의 관계도 함께 설명하세요.
5) 규정 해석에 애매함이 있을 경우, 가능한 해석을 모두 제시하세요.
6) 문서에 명시되지 않은 예외사항은 "별도 확인이 필요합니다"라고 안내하세요.
7) 답변은 5줄 이내로 간결하게 작성하세요.

<검색된_문서>
{context}
</검색된_문서>"""

        elif category == INDUSTRY_POLICY_CATEGORY:
            system_prompt = """당신은 금융업계 정책 및 법규 전문가입니다.

지침:
1) 제공된 <검색된_문서>는 금융업계 관련 법률, 정책, 규제 정보입니다.
2) 법령의 목적, 주요 내용, 적용 대상, 시행 시기 등을 체계적으로 설명하세요.
3) 금융기관에 미치는 영향과 준수해야 할 사항을 구체적으로 안내하세요.
4) 관련 법령 간의 관계나 개정 사항이 있다면 함께 설명하세요.
5) 법령 해석이 복잡한 경우, 핵심 포인트를 먼저 제시한 후 세부사항을 설명하세요.
6) 실무 적용 시 주의사항이나 예외 조건이 있다면 강조하여 안내하세요.
7) 답변은 5줄 이내로 간결하게 작성하세요.

<검색된_문서>
{context}
</검색된_문서>"""
        else:
            # 기본 시스템 메시지
            system_prompt = """당신은 KB금융그룹의 전문 AI 어시스턴트입니다.

지침:
1) 제공된 <검색된_문서>는 KB금융그룹의 공식 문서에서 검색된 정보입니다.
2) 문서 내용을 바탕으로 정확하고 구체적인 답변을 제공하세요.
3) 문서에 없는 정보는 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 안내하세요.
4) 복잡한 내용은 이해하기 쉽게 단계별로 설명하세요.
5) 답변은 5줄 이내로 간결하게 작성하세요.

<검색된_문서>
{context}
</검색된_문서>"""

        return SystemMessage(system_prompt.format(context=context_text))

# 전역 인스턴스 (싱글톤 패턴)
_langgraph_workflow = None

def get_langgraph_workflow() -> LangGraphRAGWorkflow:
    """LangGraph 워크플로우 인스턴스 반환 (싱글톤)"""
    global _langgraph_workflow
    if _langgraph_workflow is None:
        _langgraph_workflow = LangGraphRAGWorkflow()
    return _langgraph_workflow
