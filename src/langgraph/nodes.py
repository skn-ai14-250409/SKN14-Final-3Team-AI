"""
LangGraph Nodes
==============
RAG 워크플로우의 노드 함수들
"""

import logging
import time
import re
from typing import Dict, List, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

from .models import RAGState
from .session_manager import session_manager
from ..slm.slm import SLM
from .utils import (
    generate_session_title, 
    create_rag_response, 
    create_simple_response, 
    create_guardrail_response,
    extract_product_name,
    classify_product_subcategory,
    create_error_response,
    format_context,
    create_supervisor_prompt,
    get_error_message,
    get_cached_search_result,
    set_cached_search_result,
    DEFAULT_SEARCH_K,
)

# ========== Execution Path Tracking ==========

def track_execution_path(state: RAGState, node_name: str) -> RAGState:
    """실행 경로를 추적하는 헬퍼 함수"""
    execution_path = state.get("execution_path", [])
    execution_path.append(node_name)
    return {**state, "execution_path": execution_path}
    
from .utils import get_shared_slm, get_shared_vector_store

logger = logging.getLogger(__name__)

# ========== Utility Functions ==========

def clean_markdown_formatting(text: str) -> str:
    """마크다운 형식을 제거하고 일반 텍스트로 변환"""
    if not text:
        return text
    
    # 마크다운 형식 제거
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'#+\s*', '', text)             # # headers -> headers
    text = re.sub(r'`(.*?)`', r'\1', text)        # `code` -> code
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # [text](url) -> text
    text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)  # numbered lists
    
    # 불필요한 공백 정리
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 여러 줄바꿈을 두 개로 제한
    text = text.strip()
    
    return text

# ========== Node Functions ==========

def session_init_node(state: RAGState) -> RAGState:
    """세션 초기화"""
    logger.info("[NODE] session_init_node 실행 시작")
    try:
        # 실행 경로 추적
        state = track_execution_path(state, "session_init_node")
        
        # session_id 처리 (Django에서 전달받은 경우 문자열, LangGraph 내부에서는 SessionContext 객체)
        session_context_obj = state.get("session_context")
        if isinstance(session_context_obj, str):
            # Django에서 전달받은 session_id 문자열
            session_id = session_context_obj
        elif hasattr(session_context_obj, 'session_id'):
            # SessionContext 객체
            session_id = session_context_obj.session_id
        else:
            # session_id가 없는 경우
            session_id = None
        
        if not session_id:
            session_context = session_manager.create_session()
            logger.info(f"Created new session: {session_context.session_id}")
        else:
            session_context = session_manager.get_session(session_id)
            if not session_context:
                session_context = session_manager.create_session(session_id)
                logger.info(f"Recreated expired session: {session_id}")
        
        turn_id = f"turn_{int(time.time())}_{hash(str(time.time())) % 10000}"
        
        # Django에서 전달받은 conversation_history가 있으면 사용, 없으면 세션 매니저에서 로드
        django_history = state.get("conversation_history", [])
        if django_history:
            logger.info(f"[SESSION_INIT] Using Django conversation history: {len(django_history)} messages")
            conversation_history = django_history
        else:
            conversation_history = session_manager.get_conversation_history(session_context.session_id, limit=5)
        
        return {
            **state,
            "session_context": session_context,
            "turn_id": turn_id,
            "conversation_history": conversation_history,
            # 각 턴마다 초기화해야 할 필드들
            "product_name": "",
            "product_extraction_result": None,
            "category": "",
            "initial_intent": "",
            "current_topic": "",
            "active_product": "",
            "session_title": "",  # session_title 초기화
            "response": "",  # response 초기화
            "sources": [],  # sources 초기화
            "retrieved_docs": [],  # retrieved_docs 초기화
            "context_text": "",  # context_text 초기화
            "ready_to_answer": False,  # ready_to_answer 초기화
            "guardrail_decision": "",  # guardrail_decision 초기화
            "violations": [],  # violations 초기화
            "compliant_response": ""  # compliant_response 초기화
        }
        
    except Exception as e:
        logger.error(f"Session init failed: {e}")
        session_context = session_manager.create_session()
        return {
            **state,
            "session_context": session_context,
            "turn_id": f"turn_{int(time.time())}",
            "conversation_history": []
        }

def supervisor_node(state: RAGState, llm=None, slm: SLM = None) -> RAGState:
    """중앙 관리자 - 툴 선택"""
    logger.info("[NODE] supervisor_node 실행 시작")
    # 실행 경로 추적
    state = track_execution_path(state, "supervisor_node")
    
    query = state.get("query", "")
    session_context = state.get("session_context")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = get_shared_slm()
    
    # 첫 대화인지 확인 (conversation_history가 비어있을 때만)
    conversation_history = state.get("conversation_history", [])
    session_title = session_context.session_title if session_context else ""
    is_first_turn = not conversation_history or len(conversation_history) == 0
    
    # Django에서 전달받은 messages가 있으면 이를 활용하여 맥락 파악
    messages = state.get("messages", [])
    has_previous_context = messages and len(messages) > 1
    
    logger.info(f"[SUPERVISOR] Query: '{query}' | First turn: {is_first_turn} | Has previous context: {has_previous_context}")
    
    # 이전 대화 맥락이 있는 경우 이를 고려한 프롬프트 생성
    context_info = ""
    if has_previous_context:
        previous_messages = messages[:-1]  # 마지막 메시지(현재 질문) 제외
        context_parts = []
        for msg in previous_messages[-3:]:  # 최근 3개 메시지만 고려
            if hasattr(msg, 'content'):
                role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                context_parts.append(f"{role}: {msg.content[:100]}...")
        
        if context_parts:
            context_info = f"\n\n이전 대화 맥락:\n" + "\n".join(context_parts)
            logger.info(f"[SUPERVISOR] Previous context: {context_info[:200]}...")

    # 첫 번째 턴일 때: session_summary 제외한 도구들 사용 (이미 SESSION_SUMMARY 노드에서 처리됨)
    if is_first_turn:
        logger.info("[SUPERVISOR] First turn: Using tools (excluding session_summary)")
        from .tools import answer, rag_search, product_extraction
        tool_functions = [answer, rag_search, product_extraction]
    else:
        # 두 번째 턴 이후: 맥락 기반 답변 우선 고려
        logger.info("[SUPERVISOR] Multi-turn: Checking if context-based answer is possible")
        
        # 이전 대화 맥락이 충분한지 확인
        if has_previous_context:
            # 구체적인 질문인지 판단하는 키워드들
            specific_question_keywords = [
                "자격", "조건", "요건", "신청", "대상", "자격요건", "신청자격",
                "금액", "한도", "이자", "금리", "기간", "상환", "담보",
                "서류", "제출", "필요", "절차", "방법", "절차"
            ]
            
            # 구체적인 질문인지 확인
            is_specific_question = any(keyword in query for keyword in specific_question_keywords)
            
            if is_specific_question:
                logger.info("[SUPERVISOR] Specific question detected - using RAG search")
                # 구체적인 질문은 RAG 검색으로 라우팅
                from .tools import rag_search
                return {
                    **state,
                    "messages": [AIMessage(content="RAG 검색", tool_calls=[{"name": "rag_search", "args": {"query": query}, "id": f"call_{int(time.time())}_{hash(query) % 10000}"}])],
                    "ready_to_answer": False,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                }
            
            # 맥락 기반 답변이 가능한 질문인지 판단
            context_analysis_prompt = f"""
            이전 대화 맥락:
            {context_info}

            현재 질문: {query}

            위 이전 대화 내용만으로 현재 질문에 답변할 수 있는지 판단해주세요.
            답변 가능하면 "YES", 새로운 정보 검색이 필요하면 "NO"로만 응답하세요.
            """
            
            try:
                analysis_result = slm.llm.invoke(context_analysis_prompt).content.strip().upper()
                if "YES" in analysis_result:
                    logger.info("[SUPERVISOR] Context-based answer is possible - using context_answer")
                    # 맥락 기반 답변으로 직접 라우팅
                    return {
                        **state,
                        "messages": [AIMessage(content="맥락 기반 답변")],
                        "ready_to_answer": False,  # context_answer_node로 라우팅
                        "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                        "context_based": True
                    }
            except Exception as e:
                logger.warning(f"Context analysis failed: {e}, proceeding with normal tools")
        
        # 맥락 기반 답변이 불가능하거나 분석 실패 시 일반 도구 사용
        logger.info("[SUPERVISOR] Multi-turn: Using normal tools")
        from .tools import answer, rag_search, product_extraction, context_answer
        tool_functions = [answer, rag_search, product_extraction, context_answer]
    
    logger.info(f"Available tools: {[tool.name for tool in tool_functions]}")

    # 슈퍼바이저 프롬프트 생성 (새로운 3가지 노드 워크플로우에 맞게)
    supervisor_prompt = create_supervisor_prompt(
        query=query + context_info  # 이전 대화 맥락 포함
    )

    try:
        # LLM으로 툴 선택
        result = slm.llm.bind_tools(tool_functions, tool_choice="required").invoke(supervisor_prompt)
        
        # 도구 실행 결과를 state에 저장
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tool_name = result.tool_calls[0]['name']
            tool_args = result.tool_calls[0]['args']
            
            logger.info(f"[SUPERVISOR] ========= 도구 선택 완료 =========")
            logger.info(f"[SUPERVISOR] 선택된 도구: {tool_name}")
            logger.info(f"[SUPERVISOR] 도구 인자: {tool_args}")
            logger.info(f"[SUPERVISOR] 다음 노드: {tool_name.upper()}_NODE")
            
            # 도구 실행 - 새로운 워크플로우에 맞게 수정
            if tool_name == "answer":
                logger.info("[SUPERVISOR] answer 도구 실행 - FAQ 답변 생성")
                # 직접 답변 생성
                response = create_simple_response(slm, query, "faq_system")
                response = clean_markdown_formatting(response)
                return {
                    **state,
                    "messages": [result],
                    "response": response,
                    "ready_to_answer": True,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                }
            elif tool_name == "rag_search":
                logger.info("[SUPERVISOR] rag_search 도구 실행 - RAG 검색으로 라우팅")
                # RAG 검색으로 라우팅
                return {
                    **state,
                    "messages": [result],
                    "ready_to_answer": False,  # rag_search 노드로 가야 함
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                }
            elif tool_name == "product_extraction":
                logger.info("[SUPERVISOR] product_extraction 도구 실행 - 상품 추출로 라우팅")
                # product_extraction 노드로 라우팅
                return {
                    **state,
                    "messages": [result],
                    "ready_to_answer": False,  # product_extraction 노드로 가야 함
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                }
            elif tool_name == "context_answer":
                logger.info("[SUPERVISOR] context_answer 도구 실행 - 맥락 기반 답변으로 라우팅")
                # context_answer 노드로 라우팅
                return {
                    **state,
                    "messages": [result],
                    "ready_to_answer": False,  # context_answer 노드로 가야 함
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                }
        
        return {
            **state,
            "messages": [result],
            "n_tool_calling": state.get("n_tool_calling", 0) + 1,
        }
        
    except Exception as e:
        logger.error(f"Supervisor failed: {e}")
        return {
            **state,
            "messages": [AIMessage(content="죄송합니다. 처리 중 오류가 발생했습니다.")],
            "ready_to_answer": True
        }

def supervisor_router(state: RAGState) -> str:
    """슈퍼바이저 라우터"""
    logger.info("[ROUTER] supervisor_router 실행 시작")
    logger.info(f"[ROUTER] State keys: {list(state.keys())}")
    logger.info(f"[ROUTER] ready_to_answer: {state.get('ready_to_answer')}")
    
    # messages에서 마지막 메시지 확인
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        logger.info(f"[ROUTER] Last message type: {type(last_message)}")
        logger.info(f"[ROUTER] Last message content: {getattr(last_message, 'content', 'No content')}")
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info(f"[ROUTER] Tool calls: {[call['name'] for call in last_message.tool_calls]}")
    
    # RAG 검색으로 리다이렉트인지 확인
    if state.get("redirect_to_rag"):
        logger.info("[ROUTER] ========= 라우팅 결정 =========")
        logger.info("[ROUTER] redirect_to_rag=True - RAG_SEARCH로 라우팅")
        return "rag_search"
    
    # messages에서 마지막 메시지의 도구 호출 확인 (redirect_to_rag 처리 후)
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_name = last_message.tool_calls[0]['name']
            if tool_name == "rag_search":
                logger.info("[ROUTER] ========= 라우팅 결정 =========")
                logger.info("[ROUTER] 선택된 도구: rag_search")
                logger.info("[ROUTER] 다음 노드: RAG_SEARCH")
                return "rag_search"
    
    # 맥락 기반 답변인지 확인
    if state.get("context_based"):
        logger.info("[ROUTER] ========= 라우팅 결정 =========")
        logger.info("[ROUTER] context_based=True - CONTEXT_ANSWER로 라우팅")
        return "context_answer"
    
    if state.get("ready_to_answer"):
        logger.info("[ROUTER] ========= 라우팅 결정 =========")
        logger.info("[ROUTER] ready_to_answer=True - ANSWER로 라우팅")
        return "answer"
    
    # 첫 번째 턴인지 확인
    conversation_history = state.get("conversation_history", [])
    is_first_turn = not conversation_history or len(conversation_history) == 0
    
    logger.info(f"[ROUTER] conversation_history: {conversation_history}")
    logger.info(f"[ROUTER] is_first_turn: {is_first_turn}")
    
    # messages에서 마지막 메시지의 도구 호출 확인 (rag_search 제외)
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_name = last_message.tool_calls[0]['name']
            if tool_name != "rag_search":  # rag_search는 이미 위에서 처리됨
                logger.info(f"[ROUTER] ========= 라우팅 결정 =========")
                logger.info(f"[ROUTER] 선택된 도구: {tool_name}")
                logger.info(f"[ROUTER] 다음 노드: {tool_name.upper()}_NODE")
                return tool_name
    
    # 도구 호출이 없으면 기본적으로 RAG_SEARCH로
    logger.info("[ROUTER] No tool calls found - defaulting to RAG_SEARCH")
    return "rag_search"


def product_extraction_node(state: RAGState, slm: SLM = None) -> RAGState:
    """상품명 추출 노드"""
    logger.info("🏷️ [NODE] product_extraction_node 실행 시작")
    # 실행 경로 추적
    state = track_execution_path(state, "product_extraction_node")
    
    query = state.get("query", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = get_shared_slm()
    
    try:
        extracted_product = extract_product_name(slm, query)
        sub_category = classify_product_subcategory(extracted_product)
        logger.info(f"Extracted product: '{extracted_product}', Sub category: '{sub_category}'")
        
        return {
            **state,
            "product_name": extracted_product,
            "product_extraction_result": {
                "product_name": extracted_product,
                "sub_category": sub_category,
                "confidence": 0.9,
                "reasoning": f"상품명 '{extracted_product}'을 추출하고 '{sub_category}'로 분류"
            },
            "ready_to_answer": False  # 아직 답변 준비되지 않음
        }
        
    except Exception as e:
        logger.error(f"Product extraction failed: {e}")
        return {
            **state,
            "product_name": "일반",
            "ready_to_answer": False
        }

def product_search_node(state: RAGState, slm: SLM = None) -> RAGState:
    """상품 검색 노드"""
    logger.info("🔍 [NODE] product_search_node 실행 시작")
    # 실행 경로 추적
    state = track_execution_path(state, "product_search_node")
    
    query = state.get("query", "")
    product_name = state.get("product_name", "")
    
    # SLM과 VectorStore 인스턴스 생성
    if slm is None:
        slm = get_shared_slm()
    vector_store = get_shared_vector_store()
    
    try:
        # 캐시에서 검색 결과 확인
        cached_docs = get_cached_search_result(query, product_name)
        if cached_docs is not None:
            logger.info(f"Using cached search results: {len(cached_docs)} documents")
            retrieved_docs = cached_docs
        else:
            vector_store.get_index_ready()
            
            if not product_name or product_name == "일반":
                # 일반 검색
                retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K)
                logger.info(f"General search found {len(retrieved_docs)} documents")
            else:
                # 상품명으로 메타데이터 필터링 (최적화된 단일 시도)
                filter_dict = {"keywords": {"$in": [product_name]}}
                
                try:
                    retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K, filter_dict=filter_dict)
                    if retrieved_docs:
                        logger.info(f"Found {len(retrieved_docs)} documents using filter: {filter_dict}")
                    else:
                        # 필터링된 결과가 없으면 일반 검색으로 폴백
                        retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K)
                        logger.info(f"No documents found for product '{product_name}', using general search")
                        logger.info(f"General search found {len(retrieved_docs)} documents")
                except Exception as e:
                    logger.warning(f"Filter failed: {filter_dict}, error: {e}")
                    # 에러 발생 시 일반 검색으로 폴백
                    retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K)
                    logger.info(f"Fallback to general search found {len(retrieved_docs)} documents")
            
            # 검색 결과를 캐시에 저장
            set_cached_search_result(query, product_name, retrieved_docs)
        
        # PDF 정보 로깅
        if retrieved_docs:
            logger.info("📄 [PRODUCT SEARCH] 사용된 PDF 문서 정보:")
            for i, doc in enumerate(retrieved_docs):
                metadata = doc.metadata
                file_name = metadata.get('file_name', 'Unknown')
                page_number = metadata.get('page_number', 'Unknown')
                logger.info(f"  📋 문서 {i+1}: {file_name} (페이지: {page_number})")
        
        # 공통 함수를 사용하여 응답 생성
        response, sources = create_rag_response(slm, query, retrieved_docs)
        
        return {
            **state,
            "response": response,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "context_text": format_context(retrieved_docs) if retrieved_docs else "",
            "ready_to_answer": True
        }
        
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return {
            **state,
            **create_error_response("product_error")
        }

def session_summary_node(state: RAGState, slm: SLM = None) -> RAGState:
    """세션 요약 생성 노드 (첫 대화)"""
    import time
    start_time = time.time()
    logger.info("[SESSION_SUMMARY] Starting session title generation")
    # 실행 경로 추적
    state = track_execution_path(state, "session_summary_node")
    
    query = state.get("query", "")
    session_context = state.get("session_context")
    conversation_history = state.get("conversation_history", [])
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = get_shared_slm()
    
    # 첫 대화인지 확인
    conversation_history = state.get("conversation_history", [])
    is_first_turn = not conversation_history or len(conversation_history) == 0
    
    if is_first_turn:
        # 공통 함수를 사용하여 제목 생성
        session_title = generate_session_title(query, slm)
        logger.info(f"[SESSION_SUMMARY] Generated title: '{session_title}'")

        # 세션에 제목 저장
        session_manager.update_session(
            session_context.session_id,
            session_title=session_title
        )
        
        logger.info(f"Session title saved to session: {session_context.session_id}")
        
        # 업데이트된 세션 컨텍스트 가져오기
        updated_session_context = session_manager.get_session(session_context.session_id)
        if updated_session_context:
            session_context = updated_session_context
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Returning state with session_title: '{session_title}'")
        logger.info(f"📝 [NODE] session_summary_node 완료 - 실행시간: {execution_time:.2f}초")
        return {
            **state,
            "session_title": session_title,
            "ready_to_answer": False,  # RAG 검색을 위해 False로 설정
            "session_context": session_context,  # 업데이트된 세션 컨텍스트 포함
            "response": ""  # RAG 검색에서 실제 응답 생성
        }
    else:
        # 첫 대화가 아니면 기존 제목 사용하고 RAG 검색으로 넘어감
        existing_title = session_context.session_title if session_context else ""
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"📝 [NODE] session_summary_node 완료 (기존 제목 사용) - 실행시간: {execution_time:.2f}초")
        return {
            **state,
            "session_title": existing_title,
            "ready_to_answer": False,  # RAG 검색을 위해 False로 설정
            "response": ""
        }
        

def rag_search_node(state: RAGState, slm: SLM = None, vector_store=None) -> RAGState:
    """RAG 검색 노드"""
    import time
    start_time = time.time()
    logger.info("[RAG_SEARCH] Starting document search")
    # 실행 경로 추적
    state = track_execution_path(state, "rag_search_node")
    
    query = state.get("query", "")
    
    # SLM과 VectorStore 인스턴스 생성 (재사용 가능)
    if slm is None:
        slm = get_shared_slm()
    if vector_store is None:
        vector_store = get_shared_vector_store()
        vector_store.get_index_ready()  # 한 번만 초기화
    
    try:
        # 이전 대화 맥락을 고려한 검색 쿼리 생성
        messages = state.get("messages", [])
        enhanced_query = query
        
        if messages and len(messages) > 1:
            # 이전 대화 내용을 검색 쿼리에 포함
            previous_context = ""
            for msg in messages[:-1][-2:]:  # 최근 2개 메시지만 고려
                if hasattr(msg, 'content'):
                    previous_context += f" {msg.content[:50]}"
            
            if previous_context:
                enhanced_query = f"{query} {previous_context.strip()}"
                logger.info(f"[RAG_SEARCH] Enhanced query with context: {enhanced_query[:100]}...")
        
        # 문서 검색 (성능 최적화를 위해 결과 수 제한)
        retrieved_docs = vector_store.similarity_search(enhanced_query, k=3)# 더 적은 결과로 속도 향상
        logger.info(f"[RAG_SEARCH] Found {len(retrieved_docs)} documents")
        
        # 공통 함수를 사용하여 응답 생성
        response, sources = create_rag_response(slm, query, retrieved_docs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"📚 [NODE] rag_search_node 완료 - 실행시간: {execution_time:.2f}초")
        
        return {
            **state,
            "response": response,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
            "context_text": format_context(retrieved_docs) if retrieved_docs else "",
            "ready_to_answer": True
        }
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            **state,
            **create_error_response("search_error")
        }

def guardrail_check_node(state: RAGState, slm: SLM = None) -> RAGState:
    """가드레일 검사 노드"""
    logger.info("🛡️ [NODE] guardrail_check_node 실행 시작")
    # 실행 경로 추적
    state = track_execution_path(state, "guardrail_check_node")
    
    response = state.get("response", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = get_shared_slm()
    
    try:
        # 공통 함수를 사용하여 가드레일 검사
        logger.info("🛡️ [GUARDRAIL] Starting guardrail check...")
        compliant_response, violations = create_guardrail_response(slm, response)
        logger.info("🛡️ [GUARDRAIL] Guardrail check completed")
        
        # 실행 경로 로깅
        execution_path = state.get("execution_path", [])
        path_str = " -> ".join(execution_path)
        logger.info(f"🛡️ [WORKFLOW] Execution path: {path_str}")
        
        result = {
            **state,
            "guardrail_decision": "COMPLIANT" if not violations else "VIOLATION",
            "violations": violations,
            "compliant_response": compliant_response,
            "response": compliant_response,
            "ready_to_answer": True
        }
        
        logger.info("🛡️ [NODE] guardrail_check_node 완료")
        logger.info(f"🛡️ [NODE] ready_to_answer: {result.get('ready_to_answer')}")
        logger.info(f"🛡️ [NODE] response length: {len(result.get('response', ''))}")
        return result
        
    except Exception as e:
        logger.error(f"Guardrail check failed: {e}")
        result = {
            **state,
            "guardrail_decision": "ERROR",
            "violations": ["가드레일 검사 오류"],
            "compliant_response": get_error_message("guardrail_error"),
            "response": get_error_message("guardrail_error"),
            "ready_to_answer": True
        }
        logger.info("🛡️ [NODE] guardrail_check_node 완료 (에러)")
        return result

def context_answer_node(state: RAGState, slm: SLM = None) -> RAGState:
    """맥락 기반 답변 노드 - 이전 대화 내용만으로 답변"""
    import time
    start_time = time.time()
    logger.info("[CONTEXT_ANSWER] Starting context-based answer generation")
    
    # 실행 경로 추적
    state = track_execution_path(state, "context_answer_node")
    
    query = state.get("query", "")
    messages = state.get("messages", [])
    
    # SLM 인스턴스 생성
    if slm is None:
        slm = get_shared_slm()
    
    try:
        # 이전 대화 내용을 기반으로 답변 생성
        previous_context = ""
        for msg in messages[:-1]:  # 현재 질문 제외
            if hasattr(msg, 'content'):
                role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                previous_context += f"{role}: {msg.content}\n"
        
        # 맥락 기반 답변 생성 프롬프트
        context_prompt = f"""
        이전 대화 내용:
        {previous_context}

        현재 질문: {query}

        위 이전 대화 내용을 바탕으로 현재 질문에 답변해주세요. 
        이전 대화에서 이미 제공된 정보를 활용하여 자연스럽게 답변하세요.
        새로운 검색이나 문서 참조 없이 이전 대화 내용만으로 답변 가능한 경우에만 응답하세요.
        """
        
        # 맥락 기반 답변 생성
        response = slm.llm.invoke(context_prompt).content
        response = clean_markdown_formatting(response)
        
        # 답변이 일반적이거나 구체적인 정보가 부족한지 확인
        insufficient_info_keywords = [
            "일반적으로", "구체적인 조건은", "자세한 사항은", "문의하시거나", 
            "확인하는 것이 좋습니다", "다를 수 있으니", "해당 금융기관에"
        ]
        
        needs_rag_search = any(keyword in response for keyword in insufficient_info_keywords)
        
        if needs_rag_search:
            logger.info("[CONTEXT_ANSWER] Insufficient information - redirecting to RAG search")
            # 이전 대화에서 언급된 상품명 추출
            product_name = ""
            for msg in messages[:-1]:
                if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                    content = msg.content
                    # 상품명 추출 로직 (간단한 키워드 매칭)
                    if "사립학교교직원우대대출" in content:
                        product_name = "사립학교교직원우대대출"
                        break
                    elif "버팀목 전세자금대출" in content:
                        product_name = "버팀목 전세자금대출"
                        break
                    elif "햇살론" in content:
                        product_name = "햇살론"
                        break
            
            if product_name:
                logger.info(f"[CONTEXT_ANSWER] Redirecting to RAG search for product: {product_name}")
                # RAG 검색을 위한 쿼리 생성
                enhanced_query = f"{query} {product_name}"
                return {
                    **state,
                    "query": enhanced_query,
                    "messages": [AIMessage(content="RAG 검색", tool_calls=[{"name": "rag_search", "args": {"query": enhanced_query}, "id": f"call_{int(time.time())}_{hash(enhanced_query) % 10000}"}])],
                    "ready_to_answer": False,  # RAG 검색으로 리다이렉트
                    "context_based": False,
                    "redirect_to_rag": True
                }
            else:
                # 상품명을 찾을 수 없는 경우 일반 RAG 검색
                logger.info("[CONTEXT_ANSWER] No product name found - using general RAG search")
                return {
                    **state,
                    "messages": [AIMessage(content="RAG 검색", tool_calls=[{"name": "rag_search", "args": {"query": query}, "id": f"call_{int(time.time())}_{hash(query) % 10000}"}])],
                    "ready_to_answer": False,
                    "context_based": False,
                    "redirect_to_rag": True
                }
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"[CONTEXT_ANSWER] Context-based answer generated - execution time: {execution_time:.2f}s")
        
        return {
            **state,
            "response": response,
            "sources": [],  # 맥락 기반 답변이므로 소스 없음
            "ready_to_answer": True,
            "context_based": True  # 맥락 기반 답변임을 표시
        }
        
    except Exception as e:
        logger.error(f"Context answer generation failed: {e}")
        return {
            **state,
            "response": "죄송합니다. 이전 대화 내용을 바탕으로 답변을 생성할 수 없습니다.",
            "sources": [],
            "ready_to_answer": True,
            "context_based": False
        }

def answer_node(state: RAGState) -> RAGState:
    """최종 답변 노드"""
    import time
    from datetime import datetime
    start_time = time.time()
    # 로깅 최소화 (성능 개선)
    logger.debug("📝 [NODE] ANSWER_NODE 진입")
    
    # 실행 경로 추적
    state = track_execution_path(state, "answer_node")
    
    response = state.get("response", "")
    logger.info(f"📝 [ANSWER] 현재 응답 길이: {len(response)}자")
    
    # 이전 대화 맥락을 고려한 응답 개선
    messages = state.get("messages", [])
    if messages and len(messages) > 1:
        # 이전 대화 내용을 참고하여 응답 개선
        previous_context = ""
        for msg in messages[:-1][-2:]:  # 최근 2개 메시지만 고려
            if hasattr(msg, 'content'):
                role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                previous_context += f"{role}: {msg.content[:100]}...\n"
        
        if previous_context:
            logger.info(f"[ANSWER] Considering previous context: {previous_context[:200]}...")
            # 이전 대화 맥락을 고려한 응답 개선 로직 추가 가능
    
    if not response or not response.strip():
        # 기본 응답 생성 (RAG 검색이 실패한 경우에만)
        response = "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다. 다른 키워드로 다시 질문해주시거나 관련 부서에 문의해주세요."
        logger.info("[ANSWER] Using default response")
    
    # 마크다운 형식 제거
    response = clean_markdown_formatting(response)
    
    # Django에서 대화 히스토리를 처리하므로 LangGraph에서는 저장하지 않음
    logger.info("📝 [NODE] 대화 턴 저장은 Django에서 처리됩니다")
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"✅ [NODE] answer_node 완료 - 실행시간: {execution_time:.2f}초")
    
    return {
        **state,
        "response": response,
        "ready_to_answer": True
    }

