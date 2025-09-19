"""
실험용 LangGraph RAG 워크플로우

기존 orchestrator.py와 동일한 기능을 그래프 기반으로 구현
기존 코드는 그대로 유지하고 새로운 접근 방식을 시험해보기 위한 파일
"""

from typing import Dict, List, Any, TypedDict, Annotated, Optional, Pattern
import logging
import re
import time
import uuid
import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from operator import add

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.slm.slm import SLM
from src.rag.vector_store import VectorStore
from src.intent_router import IntentRouter
from src.langgraph.session_manager import session_manager, ConversationTurn, SessionContext
try:
    from src.config.keyword_mappings import (
        get_expansion_patterns,
        get_synonym_mappings,
        get_financial_terms,
        get_keyword_weights
    )
except ImportError:
    # 설정 파일이 없는 경우 기본값 사용
    def get_expansion_patterns():
        return {}
    def get_synonym_mappings():
        return {}
    def get_financial_terms():
        return {}
    def get_keyword_weights():
        return {}
from src.constants import (
    NO_ANSWER_MSG,
    MAIN_LAW, MAIN_RULE, MAIN_PRODUCT,
    GENERAL_FAQ_CATEGORY,
    COMPANY_PRODUCTS_CATEGORY,
    COMPANY_RULES_CATEGORY,
    INDUSTRY_POLICY_CATEGORY,
)

logger = logging.getLogger(__name__)

# 간단한 로깅 헬퍼 함수
def log_node_start(node_name: str, session_id: str = None):
    """노드 시작 로깅"""
    logger.info(f"[GRAPH] {node_name} started - session_id: {session_id or 'unknown'}")

def log_node_complete(node_name: str, session_id: str = None):
    """노드 완료 로깅"""
    logger.info(f"[GRAPH] {node_name} completed - session_id: {session_id or 'unknown'}")

# 공통 유틸리티 클래스
class RAGUtils:
    """RAG 관련 공통 유틸리티 메서드들"""
    
    # 공통 불용어 리스트
    STOP_WORDS = {
        "은", "는", "이", "가", "을", "를", "에", "의", "로", "으로", "도", "만", 
        "부터", "까지", "에서", "에게", "한테", "와", "과", "뭐", "있어", "있나", 
        "알려", "주세요", "안내", "정보", "어떻게", "무엇", "언제", "어디", "왜"
    }
    
    @staticmethod
    def extract_keywords_from_query(query: str) -> List[str]:
        """질문에서 핵심 키워드 추출 (최적화된 버전)"""
        try:
            # 한글, 영문, 숫자만 추출
            words = re.findall(r'[가-힣a-zA-Z0-9]+', query)
            
            # 불용어 제거 및 길이 필터링
            keywords = [word for word in words 
                       if len(word) > 1 and word not in RAGUtils.STOP_WORDS]
            
            return keywords
        except Exception as e:
            logger.error(f"[UTILS] Error extracting keywords from query: {e}")
            return []

    @staticmethod
    def extract_keywords_from_filename(filename: str) -> List[str]:
        """파일명에서 키워드 자동 추출 (최적화된 버전)"""
        try:
            # 파일명 정리 (확장자 제거, 언더스코어/공백 처리)
            clean_name = filename.replace(".pdf", "").replace("KB_", "").replace("_", " ")

            # 한글, 영문, 숫자만 추출
            keywords = re.findall(r'[가-힣a-zA-Z0-9]+', clean_name)

            # 불용어 제거 및 길이 필터링
            keywords = [kw for kw in keywords 
                       if len(kw) > 1 and kw not in RAGUtils.STOP_WORDS]

            return keywords
        except Exception as e:
            logger.error(f"[UTILS] Error extracting keywords from filename: {e}")
            return []

    @staticmethod
    def calculate_keyword_match_score(query_keywords: List[str], file_keywords: List[str]) -> float:
        """키워드 매칭 점수 계산 (통합된 버전)"""
        try:
            if not query_keywords or not file_keywords:
                return 0.0
            
            # 가중치 기반 점수 계산 사용
            return RAGUtils.calculate_weighted_keyword_score(query_keywords, file_keywords)
        except Exception as e:
            logger.error(f"[UTILS] Error calculating keyword match score: {e}")
            return 0.0

    @staticmethod
    def extract_keywords_from_product_name(product_name: str) -> List[str]:
        """상품명에서 핵심 키워드 추출 (동적 방식)"""
        keywords = []
        clean_name = product_name.replace("KB", "").strip()
        
        # 1. 기본 단어 추출
        words = re.findall(r'[가-힣]+', clean_name)
        for word in words:
            if len(word) > 1:
                keywords.append(word)
        
        # 2. 동적 키워드 확장 (하드코딩 대신 패턴 기반)
        expanded_keywords = RAGUtils._expand_keywords_dynamically(keywords, product_name)
        keywords.extend(expanded_keywords)
        
        return list(set(keywords))
    
    @staticmethod
    def _expand_keywords_dynamically(base_keywords: List[str], product_name: str) -> List[str]:
        """동적으로 키워드를 확장하는 메서드 (설정 파일 기반)"""
        expanded = []
        
        # 설정 파일에서 패턴 가져오기
        expansion_patterns = get_expansion_patterns()
        synonym_mappings = get_synonym_mappings()
        
        # 1. 패턴 기반 키워드 확장
        for pattern, related_terms in expansion_patterns.items():
            if re.search(pattern, product_name):
                expanded.extend(related_terms)
        
        # 2. 유사어/동의어 확장
        for keyword in base_keywords:
            if keyword in synonym_mappings:
                expanded.extend(synonym_mappings[keyword])
        
        # 3. 금융 도메인 특화 키워드 추가
        financial_terms = get_financial_terms()
        for keyword in base_keywords:
            if keyword in financial_terms:
                expanded.extend(financial_terms[keyword])
        
        return expanded
    
    @staticmethod
    def calculate_weighted_keyword_score(query_keywords: List[str], file_keywords: List[str]) -> float:
        """가중치를 적용한 키워드 매칭 점수 계산"""
        try:
            if not query_keywords or not file_keywords:
                return 0.0
            
            keyword_weights = get_keyword_weights()
            total_score = 0.0
            matched_count = 0
            
            for q_kw in query_keywords:
                # 정확한 매칭
                if q_kw in file_keywords:
                    weight = keyword_weights.get(q_kw, 0.5)  # 기본 가중치 0.5
                    total_score += weight * 1.0
                    matched_count += 1
                
                # 부분 매칭
                for f_kw in file_keywords:
                    if q_kw in f_kw or f_kw in q_kw:
                        weight = keyword_weights.get(q_kw, 0.3)  # 부분 매칭은 낮은 가중치
                        total_score += weight * 0.3
                        matched_count += 1
                        break  # 중복 계산 방지
            
            # 정규화 (매칭된 키워드 수로 나누기)
            return total_score / len(query_keywords) if query_keywords else 0.0
            
        except Exception as e:
            logger.error(f"[UTILS] Error calculating weighted keyword score: {e}")
            return 0.0

    @staticmethod
    def normalize_retrieved(items) -> List[Document]:
        """검색 결과 정규화"""
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

    @staticmethod
    def format_context(docs) -> str:
        """컨텍스트 포맷"""
        lines = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("relative_path") or meta.get("file_name") or meta.get("source_type") or "unknown_source"
            snippet = (d.page_content or "").strip()
            if not snippet:
                continue
            lines.append(f"[source: {src}]\n{snippet}")
        return "\n---\n".join(lines)

    @staticmethod
    def filter_by_relevance_score(docs: List[Document], query: str) -> List[Document]:
        """관련성 필터링"""
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

# --------------------- Guardrail YAML dataclasses ---------------------
@dataclass
class PolicySourceRef:
    file: str
    clause: Optional[str] = None

@dataclass
class PolicyRule:
    rule_id: str
    policy: str
    severity: str                 # "HIGH" | "MEDIUM" | "LOW"
    patterns: List[str]
    disclosures: List[str] = field(default_factory=list)
    fix_hint: str = ""
    sources: List[PolicySourceRef] = field(default_factory=list)
    compiled: List[Pattern] = field(default_factory=list)

@dataclass
class SoftFixRule:
    pattern: str
    replacement: str
    compiled: Optional[Pattern] = None

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
    session_context: SessionContext  # 멀티턴 세션 컨텍스트
    conversation_history: List[ConversationTurn]  # 대화 히스토리
    turn_id: str  # 현재 턴 ID
    # guardrail 결과 (최소)
    guardrail_decision: str
    violations: List[Dict[str, Any]]
    compliant_response: str

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
        # Guardrail 자료구조 (YAML 로드 후 사용)
        self._policy_rules: List[PolicyRule] = []
        self._soft_fix_rules: List[SoftFixRule] = []
        self._glossary_terms: List[Dict[str, str]] = []
        self._glossary_regex_terms: List[Dict[str, str]] = []
        self._glossary_opts: Dict[str, Any] = {}
        self._glossary_regex_compiled: List[re.Pattern] = []

        self.workflow = self._build_workflow()
        # 성능 최적화를 위한 캐시
        self._filename_cache = None
        self._filename_index = None

        # 최소 YAML 로드
        # 실제 guardrails 폴더의 policy_rules.yaml 경로로 수정
        self._load_minimal_guardrail_yamls("src/langgraph/guardrails/policy_rules.yaml")

    def _build_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 구축 (멀티턴 대화 지원)"""
        workflow = StateGraph(RAGState)

        # 노드 추가
        workflow.add_node("session_init", self._session_init)  # 세션 초기화
        workflow.add_node("context_analysis", self._context_analysis)  # 맥락 분석
        workflow.add_node("first_turn_preprocess", self._first_turn_preprocess)  # 첫 턴 전처리
        workflow.add_node("classify_intent", self._classify_intent)
        workflow.add_node("handle_general_faq", self._handle_general_faq)
        workflow.add_node("extract_product_name", self._extract_product_name)
        workflow.add_node("search_documents", self._search_documents)
        workflow.add_node("filter_relevance", self._filter_relevance)
        workflow.add_node("generate_response", self._generate_response)
        # 🔹 최종 가드레일 노드 추가
        workflow.add_node("guardrails", self._guardrails_slim_inline)
        workflow.add_node("save_conversation", self._save_conversation)  # 대화 저장

        # 엣지 추가 (단순화된 플로우)
        workflow.add_edge(START, "session_init")
        workflow.add_edge("session_init", "first_turn_preprocess")  # context_analysis 우회
        workflow.add_edge("first_turn_preprocess", "classify_intent")

        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_category,
            {
                "general_faq": "handle_general_faq",
                "rag_needed": "extract_product_name"
            }
        )

        workflow.add_edge("handle_general_faq", "save_conversation")
        workflow.add_edge("extract_product_name", "search_documents")
        workflow.add_edge("search_documents", "filter_relevance")
        workflow.add_edge("filter_relevance", "generate_response")
        # 🔹 generate_response → guardrails → save_conversation
        workflow.add_edge("generate_response", "guardrails")
        workflow.add_edge("guardrails", "save_conversation")
        workflow.add_edge("save_conversation", END)

        return workflow.compile()

    # ------------------------- Guardrail: YAML Loader -------------------------
    def _load_minimal_guardrail_yamls(self, policy_path: str = "config/policy_rules.yaml"):
        """policy_rules.yaml + glossary_terms.yaml 로드해서
        _guardrails_slim_inline이 바로 쓸 수 있게 셋업."""
        self._policy_rules = []
        self._soft_fix_rules = []
        self._glossary_terms = []
        self._glossary_regex_terms = []
        self._glossary_opts = {}
        self._glossary_regex_compiled = []

        if not os.path.exists(policy_path):
            logger.warning(f"[GUARD] policy file not found: {policy_path}")
            return

        with open(policy_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # rules
        for r in data.get("rules", []) or []:
            srcs = [PolicySourceRef(**s) for s in (r.get("sources") or [])]
            rule = PolicyRule(
                rule_id=r["rule_id"],
                policy=r.get("policy", "INTERNAL"),
                severity=r.get("severity", "MEDIUM").upper(),
                patterns=r.get("patterns", []),
                disclosures=r.get("disclosures", []) or [],
                fix_hint=r.get("fix_hint", "") or "",
                sources=srcs
            )
            rule.compiled = [re.compile(p, flags=re.IGNORECASE) for p in rule.patterns]
            self._policy_rules.append(rule)

        # soft_fixes
        for sf in data.get("soft_fixes", []) or []:
            s = SoftFixRule(pattern=sf["pattern"], replacement=sf["replacement"])
            s.compiled = re.compile(s.pattern, flags=re.IGNORECASE)
            self._soft_fix_rules.append(s)

        # glossary
        glossary_file = ((data.get("terminology") or {}).get("normalization") or {}).get("glossary_file")
        # 경로가 상대경로면 guardrails 폴더 기준으로 보정
        if glossary_file and not os.path.isabs(glossary_file):
            glossary_file = os.path.join(os.path.dirname(policy_path), os.path.basename(glossary_file))
        if glossary_file and os.path.exists(glossary_file):
            with open(glossary_file, "r", encoding="utf-8") as gf:
                g = yaml.safe_load(gf) or {}
            self._glossary_terms = g.get("terms") or []
            self._glossary_regex_terms = g.get("regex_terms") or []
            self._glossary_opts = g.get("options") or {}
            flags = re.IGNORECASE if self._glossary_opts.get("case_insensitive", True) else 0
            self._glossary_regex_compiled = [
                re.compile(item["pattern"], flags=flags) for item in (self._glossary_regex_terms or [])
            ]

        logger.info(f"[GUARD] loaded: rules={len(self._policy_rules)}, soft_fixes={len(self._soft_fix_rules)}, glossary_terms={len(self._glossary_terms)}")

    # ------------------------- Guardrail: Node (inline) -------------------------
    def _guardrails_slim_inline(self, state: RAGState) -> RAGState:
        """
        최종 가드레일(인라인 버전: 헬퍼 호출 없음)
          - 규칙(YAML: self._policy_rules)으로 응답 검사 → HIGH 있으면 BLOCK
          - 소프트 치환(self._soft_fix_rules) 적용
          - 용어 표준화(self._glossary_terms / self._glossary_regex_compiled) 적용
          - 출처/정확성/누락/강조 등은 수행하지 않음
        """
        # (선택) 로더를 이미 호출했으므로 추가 리로드는 생략

        resp = (state.get("response") or "").strip()
        if not resp:
            return {
                **state,
                "guardrail_decision": "PASS",
                "violations": [],
                "compliant_response": state.get("response", "")
            }

        # 1) 규칙 매칭
        violations = []
        for rule in (self._policy_rules or []):
            for pat in (rule.compiled or []):
                m = pat.search(resp)
                if not m:
                    continue
                violations.append({
                    "phase": "post",
                    "policy": getattr(rule, "policy", "INTERNAL"),
                    "rule_id": getattr(rule, "rule_id", "UNKNOWN"),
                    "severity": getattr(rule, "severity", "MEDIUM"),
                    "evidence": m.group(0),
                    "fix_hint": getattr(rule, "fix_hint", ""),
                    "sources": [
                        {"file": s.file, "clause": s.clause}
                        for s in (getattr(rule, "sources", []) or [])
                    ],
                })

        # HIGH 위반 → BLOCK
        if any(v.get("severity") == "HIGH" for v in violations):
            safe = (
                "죄송하지만 해당 질문은 내부 기준상 구체적으로 답변드리기 어렵습니다."
            )
            return {
                **state,
                "guardrail_decision": "BLOCK",
                "violations": violations,
                "response": safe,
                "compliant_response": safe,
                "sources": state.get("sources", [])
            }

        # 2) 소프트 치환
        fixed = resp
        soft_changed = False
        for sf in (self._soft_fix_rules or []):
            try:
                compiled = getattr(sf, "compiled", None)
                replacement = getattr(sf, "replacement", "")
                if compiled and compiled.search(fixed):
                    fixed = compiled.sub(replacement, fixed)
                    soft_changed = True
            except Exception:
                continue

        # 3) 용어 표준화
        # 3-1) terms: from → to
        terms = self._glossary_terms or []
        if terms:
            ci = bool((self._glossary_opts or {}).get("case_insensitive", True))
            wb = bool((self._glossary_opts or {}).get("word_boundary", True))
            flags = re.IGNORECASE if ci else 0
            for t in terms:
                src = t.get("from")
                dst = t.get("to")
                if not src or not dst:
                    continue
                pat = re.escape(src)
                if wb:
                    pat = rf"\b{pat}\b"
                try:
                    fixed_new = re.sub(pat, dst, fixed, flags=flags)
                    if fixed_new != fixed:
                        fixed = fixed_new
                        soft_changed = True
                except Exception:
                    continue

        # 3-2) regex_terms: pattern → replacement
        regex_compiled = self._glossary_regex_compiled or []
        regex_terms = self._glossary_regex_terms or []
        if regex_compiled and regex_terms:
            for idx, pat in enumerate(regex_compiled):
                try:
                    repl = ""
                    if idx < len(regex_terms):
                        repl = regex_terms[idx].get("replacement", "") or ""
                    fixed_new = pat.sub(repl, fixed)
                    if fixed_new != fixed:
                        fixed = fixed_new
                        soft_changed = True
                except Exception:
                    continue

        decision = "SOFT_FIX" if (soft_changed or violations) else "PASS"

        return {
            **state,
            "guardrail_decision": decision,
            "violations": violations,
            "response": fixed,
            "compliant_response": fixed,
        }

    # ------------------------- 기존 로직 -------------------------
    def _session_init(self, state: RAGState) -> RAGState:
        """세션 초기화 및 관리"""
        try:
            session_id = state.get("session_context", {}).session_id if state.get("session_context") else None
            log_node_start("session_init", session_id)
            
            if not session_id:
                # 새 세션 생성
                session_context = session_manager.create_session()
                logger.info(f"[GRAPH] Created new session: {session_context.session_id}")
            else:
                # 기존 세션 조회
                session_context = session_manager.get_session(session_id)
                if not session_context:
                    # 세션이 만료되었거나 없으면 새로 생성
                    session_context = session_manager.create_session(session_id)
                    logger.info(f"[GRAPH] Recreated expired session: {session_id}")
                else:
                    logger.info(f"[GRAPH] Retrieved existing session: {session_id}")
            
            # 턴 ID 생성
            turn_id = f"turn_{int(time.time())}_{hash(str(time.time())) % 10000}"
            
            result = {
                **state,
                "session_context": session_context,
                "turn_id": turn_id,
                "conversation_history": session_manager.get_conversation_history(session_context.session_id, limit=5)
            }
            log_node_complete("session_init", session_context.session_id)
            return result
            
        except Exception as e:
            logger.error(f"[GRAPH] Session init failed: {e}")
            # 폴백: 기본 세션 생성
            session_context = session_manager.create_session()
            return {
                **state,
                "session_context": session_context,
                "turn_id": f"turn_{int(time.time())}",
                "conversation_history": []
            }

    def _context_analysis(self, state: RAGState) -> RAGState:
        """대화 맥락 분석 및 라우팅 결정"""
        try:
            session_context = state.get("session_context")
            conversation_history = state.get("conversation_history", [])
            query = state.get("query", "")
            
            logger.info(f"[GRAPH] Context analysis - session_id: {session_context.session_id if session_context else 'None'}, conversation_history length: {len(conversation_history)}")
            
            if not session_context:
                logger.info(f"[GRAPH] No session context - routing to first_turn")
                return {**state, "context_route": "first_turn"}
            
            # 첫 턴인지 확인
            if not conversation_history:
                logger.info(f"[GRAPH] First turn detected for session: {session_context.session_id}")
                return {**state, "context_route": "first_turn"}
            else:
                logger.info(f"[GRAPH] Continue turn - conversation history exists: {len(conversation_history)} turns")
                logger.info(f"[GRAPH] Conversation history content: {[turn.turn_id for turn in conversation_history]}")
            
            # 임시로 항상 first_turn으로 라우팅 (디버깅용)
            logger.info(f"[GRAPH] FORCE ROUTING TO FIRST_TURN FOR DEBUGGING")
            return {**state, "context_route": "first_turn"}
            
            # 이전 대화 맥락 분석
            last_turn = conversation_history[-1]
            
            # 맥락 기반 의도 분석
            context_intent = self._analyze_context_intent(query, last_turn, session_context)
            
            # 세션 컨텍스트 업데이트
            session_manager.update_session(
                session_context.session_id,
                current_topic=self._extract_current_topic(query, last_turn),
                conversation_summary=session_manager.generate_conversation_summary(session_context.session_id)
            )
            
            logger.info(f"[GRAPH] Context analysis completed. Route: continue_turn, Intent: {context_intent}")
            return {
                **state,
                "context_route": "continue_turn",
                "context_intent": context_intent
            }
            
        except Exception as e:
            logger.error(f"[GRAPH] Context analysis failed: {e}")
            return {**state, "context_route": "first_turn"}

    def _analyze_context_intent(self, query: str, last_turn: ConversationTurn, session_context: SessionContext) -> str:
        """맥락 기반 의도 분석"""
        try:
            # 이전 대화와의 연관성 분석
            context_prompt = f"""
            이전 대화:
            Q: {last_turn.user_query}
            A: {last_turn.ai_response[:200]}...
            
            현재 질문: {query}
            
            현재 질문이 이전 대화와 어떤 관련이 있는지 분석해주세요:
            1. follow_up: 이전 질문에 대한 추가 질문이나 세부사항
            2. related: 같은 주제의 새로운 질문
            3. new_topic: 완전히 새로운 주제
            4. clarification: 이전 답변에 대한 명확화 요청
            
            답변 형식: [의도] - [간단한 설명]
            """
            
            response = self.slm.generate_response(context_prompt)
            
            if "follow_up" in response.lower():
                return "follow_up"
            elif "related" in response.lower():
                return "related"
            elif "new_topic" in response.lower():
                return "new_topic"
            else:
                return "clarification"
                
        except Exception as e:
            logger.error(f"[GRAPH] Context intent analysis failed: {e}")
            return "related"

    def _extract_current_topic(self, query: str, last_turn: ConversationTurn) -> str:
        """현재 토픽 추출"""
        try:
            # 간단한 키워드 기반 토픽 추출
            keywords = RAGUtils.extract_keywords_from_query(query)
            if keywords:
                return " ".join(keywords[:3])  # 상위 3개 키워드
            
            # 이전 토픽과의 연관성 확인
            if last_turn.product_name:
                return last_turn.product_name
            
            return query[:50] + "..." if len(query) > 50 else query
            
        except Exception as e:
            logger.error(f"[GRAPH] Topic extraction failed: {e}")
            return query[:30] + "..." if len(query) > 30 else query

    def _route_by_context(self, state: RAGState) -> str:
        """맥락 기반 라우팅"""
        return state.get("context_route", "first_turn")

    def _save_conversation(self, state: RAGState) -> RAGState:
        """대화 턴 저장"""
        try:
            session_context = state.get("session_context")
            turn_id = state.get("turn_id")
            
            if not session_context or not turn_id:
                return state
            
            # 대화 턴 생성
            conversation_turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=datetime.now(),
                user_query=state.get("query", ""),
                ai_response=state.get("response", ""),
                category=state.get("category", ""),
                product_name=state.get("product_name", ""),
                sources=state.get("sources", []),
                session_context=session_context.to_dict()
            )
            
            # 세션 매니저에 저장
            session_manager.add_conversation_turn(session_context.session_id, conversation_turn)
            
            # 메시지 히스토리에 추가
            session_manager.add_message(session_context.session_id, HumanMessage(content=state.get("query", "")))
            session_manager.add_message(session_context.session_id, AIMessage(content=state.get("response", "")))
            
            logger.info(f"[GRAPH] Saved conversation turn: {turn_id}")
            return state
            
        except Exception as e:
            logger.error(f"[GRAPH] Save conversation failed: {e}")
            return state

    def _first_turn_preprocess(self, state: RAGState) -> RAGState:
        """세션의 첫 질문에 대해서만 실행되는 전처리 노드(강화된 예외 처리).

        동작:
        - 세션 첫 질문인지 확인 (conversation_history)
        - router로 intent 획득(initial_intent) — 안전한 예외 처리
        - SLM으로 챗봇 세션용 짧은 제목 생성(initial_topic_summary)
          (SLM 호출 실패에 대해서만 최소 폴백 처리)
        """
        # 세션 컨텍스트 확인
        session_context = state.get("session_context")
        conversation_history = state.get("conversation_history", [])
        
        logger.info(f"[GRAPH] First turn preprocess - conversation_history length: {len(conversation_history)}")
        
        # 첫 턴인지 확인 (conversation_history가 비어있거나 현재 턴이 첫 번째인 경우)
        is_first_turn = not conversation_history or len(conversation_history) == 0
        
        if not is_first_turn:
            logger.info(f"[GRAPH] Skipping first turn preprocess - conversation already exists")
            return state

        query = state.get("query", "") or ""
        
        # 제목 생성: LLM 호출 (최대 15자 제한)
        logger.info(f"[GRAPH] FIRST_TURN_PREPROCESS EXECUTING - Query: {query[:50]}...")
        
        system_message = SystemMessage(
            "당신은 '챗봇 세션 제목 생성기'입니다. 사용자의 질문을 보고 "
            "이 세션을 대표할 매우 간결한 한 줄 제목을 생성하세요."
        )
        human_prompt = (
            f"사용자 질문: {query}\n\n"
            "출력: 제목 텍스트만 한 줄로 출력\n"
            "- 한국어로 간결하게 (최대 15자)\n"
            "- 불필요한 설명 없이 제목만 반환\n"
        )
        messages = [system_message, HumanMessage(human_prompt)]

        session_title = ""
        MAX_TITLE_CHARS = 15  # 15자로 제한

        try:
            logger.info(f"[GRAPH] Generating title for query: {query[:50]}...")
            raw = (self.slm.invoke(messages) or "").strip()
            logger.info(f"[GRAPH] Raw title response: {raw[:100]}...")
            if raw:
                session_title = raw.splitlines()[0].strip()
                if len(session_title) > MAX_TITLE_CHARS:
                    session_title = session_title[:MAX_TITLE_CHARS].rstrip()
                logger.info(f"[GRAPH] Generated title: {session_title}")
            else:
                logger.warning(f"[GRAPH] Empty title response from SLM")
        except Exception as e:
            # 최소 폴백: intent 또는 질문의 앞부분 사용
            logger.warning(f"[GRAPH] first_turn_preprocess: title generation failed: {e}")
            session_title = ""

        if not session_title:
            # LLM 실패 또는 빈 결과일 때 폴백: intent 우선, 없으면 질문 앞부분
            session_title = initial_intent or (query.strip().splitlines()[0][:MAX_TITLE_CHARS] or "새로운 질문")
            logger.info(f"[GRAPH] Using fallback title: {session_title}")
        
        # 최종 보장: 제목이 절대 비어있지 않도록
        if not session_title or session_title.strip() == "":
            session_title = "새로운 질문"
            logger.info(f"[GRAPH] Final fallback title: {session_title}")
        
        # 15자 제한 적용
        if len(session_title) > MAX_TITLE_CHARS:
            session_title = session_title[:MAX_TITLE_CHARS].rstrip()
        
        
        # Router 호출 - 안전한 예외 처리
        try:
            initial_intent = self.router.route_prompt(query)
            if not initial_intent or not initial_intent.strip():
                logger.warning(f"[GRAPH] Router returned empty intent for query: {query[:100]}...")
                initial_intent = "unknown"
            else:
                initial_intent = initial_intent.strip()
        except Exception as e:
            logger.error(f"[GRAPH] Router failed for query '{query[:100]}...': {e}")
            initial_intent = "unknown"

        # 세션 컨텍스트 업데이트
        if session_context:
            session_manager.update_session(
                session_context.session_id,
                initial_intent=initial_intent,
                session_title=session_title
            )
        
        logger.info(
            f"[GRAPH] First-turn preprocess done. session_id={session_context.session_id if session_context else 'unknown'}, "
            f"intent={initial_intent!r}, session_title={session_title!r}"
        )

        return {
            **state,
            "initial_intent": initial_intent,
            "initial_topic_summary": session_title,
        }

    def _classify_intent(self, state: RAGState) -> RAGState:
        """인텐트 분류 노드 - 강화된 에러 처리 (제목 생성 포함)"""
        query = state["query"]
        session_context = state.get("session_context")
        conversation_history = state.get("conversation_history", [])
        
        try:
            category = self.router.route_prompt(query)
            if not category or not category.strip():
                logger.warning(f"[GRAPH] Router returned empty category for query: {query[:100]}...")
                category = "unknown"
            else:
                category = category.strip()
        except Exception as e:
            logger.error(f"[GRAPH] Intent classification failed for query '{query[:100]}...': {e}")
            category = "unknown"

        # 첫 턴인 경우 제목 생성은 run_workflow에서 처리
        session_title = ""

        logger.info(f"[GRAPH] Classified category: {category}")

        return {
            **state,
            "category": category,
            "initial_intent": category,
            "initial_topic_summary": session_title
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
        """일반 FAQ 처리 노드 - 강화된 에러 처리"""
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
        
        try:
            response = self.slm.invoke(messages)
            if not response or not response.strip():
                logger.warning(f"[GRAPH] SLM returned empty response for general FAQ")
                response = "죄송합니다. 현재 답변을 생성할 수 없습니다. 다시 시도해 주세요."
        except Exception as e:
            logger.error(f"[GRAPH] SLM failed for general FAQ: {e}")
            response = "죄송합니다. 현재 답변을 생성할 수 없습니다. 다시 시도해 주세요."

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
            query_keywords = RAGUtils.extract_keywords_from_query(query)
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
                    keywords = RAGUtils.extract_keywords_from_product_name(product_name)
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
        retrieved_docs = RAGUtils.normalize_retrieved(raw_retrieved)

        logger.info(f"[GRAPH] Retrieved {len(retrieved_docs)} documents")

        return {
            **state,
            "retrieved_docs": retrieved_docs
        }

    def _find_exact_filename_match(self, query_keywords: List[str]) -> str:
        """질문 키워드로 정확한 파일명 찾기 (단순화된 방식)"""
        try:
            # 캐시된 파일명 인덱스 사용
            if self._filename_index is None:
                self._build_filename_index()

            if not self._filename_index:
                return ""

            best_match = ""
            max_score = 0.0
            min_threshold = 0.3

            # 간단한 매칭 로직
            for filename, file_keywords in self._filename_index.items():
                score = RAGUtils.calculate_keyword_match_score(query_keywords, file_keywords)
                if score > max_score and score >= min_threshold:
                    max_score = score
                    best_match = filename

            return best_match if max_score >= min_threshold else ""

        except Exception as e:
            logger.error(f"[GRAPH] Error in filename matching: {e}")
            return ""

    def _build_filename_index(self) -> None:
        """파일명 인덱스 구축 (단순화된 버전)"""
        try:
            available_files = self.vector_store.get_available_files()
            if not available_files:
                self._filename_index = {}
                return

            self._filename_index = {
                filename: RAGUtils.extract_keywords_from_filename(filename)
                for filename in available_files
            }

        except Exception as e:
            logger.error(f"[GRAPH] Error building filename index: {e}")
            self._filename_index = {}

# ----------------------------------------------------------------------------------------------------------------------------

    def _filter_relevance(self, state: RAGState) -> RAGState:
        """관련성 필터링 노드"""
        docs = state["retrieved_docs"]
        query = state["query"]

        filtered_docs = RAGUtils.filter_by_relevance_score(docs, query)
        context_text = RAGUtils.format_context(filtered_docs)

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

    def run_workflow(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """멀티턴 대화 지원 워크플로우 실행"""
        try:
            # 기존 세션 조회 또는 새 세션 생성
            if session_id:
                session_context = session_manager.get_session(session_id)
                if not session_context:
                    session_context = session_manager.create_session(session_id)
            else:
                session_context = session_manager.create_session()
            
            # 초기 상태 설정 (멀티턴 지원)
            initial_state = {
                "messages": [],
                "query": query,
                "category": "",
                "product_name": "",
                "retrieved_docs": [],
                "context_text": "",
                "response": "",
                "sources": [],
                "session_context": session_context,
                "conversation_history": [],
                "turn_id": "",
                "guardrail_decision": "",
                "violations": [],
                "compliant_response": "",
            }

            # 워크플로우 실행
            final_state = self.workflow.invoke(initial_state)

            # 세션 컨텍스트 업데이트
            updated_session = final_state.get("session_context")
            if updated_session:
                session_manager.update_session(
                    updated_session.session_id,
                    last_activity=datetime.now()
                )

            # 제목 생성 (첫 턴인 경우)
            conversation_history = session_manager.get_conversation_history(session_context.session_id)
            initial_topic_summary = final_state.get("initial_topic_summary", "")
            
            # LLM으로 제목 생성 (첫 턴인 경우)
            if not conversation_history and not initial_topic_summary:
                logger.info(f"[GRAPH] Generating title with LLM for first turn")
                
                try:
                    # LLM으로 제목 생성
                    system_message = SystemMessage(
                        "당신은 '챗봇 세션 제목 생성기'입니다. 사용자의 질문을 보고 "
                        "이 세션을 대표할 매우 간결한 한 줄 제목을 생성하세요."
                    )
                    human_prompt = (
                        f"사용자 질문: {query}\n\n"
                        "출력: 제목 텍스트만 한 줄로 출력\n"
                        "- 한국어로 간결하게 (반드시 15자 이하)\n"
                        "- 불필요한 설명 없이 제목만 반환\n"
                        "- 예시: '햇살론 문의', '대출 문의', '상품 안내'\n"
                        "- 중요: 15자를 초과하면 안됩니다!\n"
                    )
                    messages = [system_message, HumanMessage(human_prompt)]
                    
                    raw_title = (self.slm.invoke(messages) or "").strip()
                    if raw_title:
                        initial_topic_summary = raw_title.splitlines()[0].strip()
                        # 15자 제한 강제 적용
                        if len(initial_topic_summary) > 15:
                            initial_topic_summary = initial_topic_summary[:15].rstrip()
                            logger.info(f"[GRAPH] Title truncated to 15 chars: {initial_topic_summary}")
                    else:
                        # LLM 실패시 폴백
                        initial_topic_summary = query[:15] if len(query) > 15 else query
                    
                    # 세션에 제목 저장
                    session_manager.update_session(
                        session_context.session_id,
                        session_title=initial_topic_summary
                    )
                    
                    logger.info(f"[GRAPH] LLM generated title: {initial_topic_summary}")
                    
                except Exception as e:
                    logger.error(f"[GRAPH] LLM title generation failed: {e}")
                    # 폴백: 질문 앞부분 사용
                    initial_topic_summary = query[:15] if len(query) > 15 else query
                    session_manager.update_session(
                        session_context.session_id,
                        session_title=initial_topic_summary
                    )

            # 응답 구성
            response_data = {
                "response": final_state["response"],
                "sources": final_state["sources"],
                "category": final_state.get("category", "unknown"),
                "product_name": final_state.get("product_name", ""),
                "session_info": {
                    "session_id": updated_session.session_id if updated_session else session_context.session_id,
                    "initial_intent": final_state.get("initial_intent", ""),
                    "initial_topic_summary": initial_topic_summary,
                    "conversation_mode": updated_session.conversation_mode if updated_session else "normal",
                    "current_topic": updated_session.current_topic if updated_session else "",
                    "active_product": updated_session.active_product if updated_session else "",
                },
                # 호환성 유지
                "initial_intent": final_state.get("initial_intent", ""),
                "initial_topic_summary": initial_topic_summary,
                "guardrail": {
                    "decision": final_state.get("guardrail_decision", ""),
                    "violations": final_state.get("violations", []),
                }
            }
            
            logger.info(f"[GRAPH] Workflow completed for session: {session_context.session_id}")
            return response_data
            
        except Exception as e:
            logger.error(f"[GRAPH] Workflow execution failed: {e}")
            return {
                "response": "처리 중 오류가 발생했습니다.",
                "sources": [],
                "category": "error",
                "product_name": "",
                "session_info": {
                    "session_id": session_id or "error",
                    "initial_intent": "",
                    "initial_topic_summary": "",
                    "conversation_mode": "error",
                    "current_topic": "",
                    "active_product": "",
                },
                "initial_intent": "",
                "initial_topic_summary": "",
                "guardrail": {
                    "decision": "error",
                    "violations": [],
                }
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
