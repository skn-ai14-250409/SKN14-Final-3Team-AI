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
    DEFAULT_SEARCH_K,
    ERROR_MESSAGES
)
from ..slm.slm import SLM
from ..rag.vector_store import VectorStore

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
        session_id = state.get("session_context", {}).session_id if state.get("session_context") else None
        
        if not session_id:
            session_context = session_manager.create_session()
            logger.info(f"Created new session: {session_context.session_id}")
        else:
            session_context = session_manager.get_session(session_id)
            if not session_context:
                session_context = session_manager.create_session(session_id)
                logger.info(f"Recreated expired session: {session_id}")
        
        turn_id = f"turn_{int(time.time())}_{hash(str(time.time())) % 10000}"
        
        return {
            **state,
            "session_context": session_context,
            "turn_id": turn_id,
            "conversation_history": session_manager.get_conversation_history(session_context.session_id, limit=5),
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
    query = state.get("query", "")
    session_context = state.get("session_context")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
    # 첫 대화인지 확인 (conversation_history가 비어있을 때만)
    conversation_history = state.get("conversation_history", [])
    session_title = session_context.session_title if session_context else ""
    is_first_turn = not conversation_history or len(conversation_history) == 0
    
    logger.info(f"[SUPERVISOR] Query: '{query}' | First turn: {is_first_turn}")

    # 첫 번째 턴일 때: session_summary 제외한 도구들 사용 (이미 SESSION_SUMMARY 노드에서 처리됨)
    if is_first_turn:
        logger.info("[SUPERVISOR] First turn: Using tools (excluding session_summary)")
        from .tools import general_faq, rag_search, product_extraction, product_search, answer
        tool_functions = [general_faq, rag_search, product_extraction, product_search, answer]
    else:
        # 두 번째 턴 이후: session_summary, intent_classification 제외한 도구들만 사용
        logger.info("[SUPERVISOR] Multi-turn: Using limited tools")
        from .tools import general_faq, rag_search, product_extraction, product_search, answer
        tool_functions = [general_faq, rag_search, product_extraction, product_search, answer]
    
    logger.info(f"Available tools: {[tool.name for tool in tool_functions]}")

    # 이미 분류된 의도 사용 (intent_classification_node에서 설정됨)
    intent_category = state.get("intent_category", "general_banking_FAQs")
    
    # 응답이 이미 생성되었는지 확인
    current_response = state.get("response", "")
    has_response = bool(current_response.strip())
    
    # 상품명이 추출되었는지 확인
    extracted_product = state.get("product_name", "")
    has_product_name = bool(extracted_product.strip()) and extracted_product != "일반"
    
    # 슈퍼바이저 프롬프트 생성
    supervisor_prompt = create_supervisor_prompt(
        query=query,
        is_first_turn=is_first_turn,
        intent_category=intent_category,
        has_response=has_response,
        extracted_product=extracted_product,
        has_product_name=has_product_name
    )

    try:
        # LLM으로 툴 선택
        result = slm.llm.bind_tools(tool_functions, tool_choice="required").invoke(supervisor_prompt)
        
        # 도구 실행 결과를 state에 저장
        if hasattr(result, 'tool_calls') and result.tool_calls:
            tool_name = result.tool_calls[0]['name']
            tool_args = result.tool_calls[0]['args']
            
            logger.info(f"[SUPERVISOR] Selected tool: {tool_name} with args: {tool_args}")
            
            # 도구 실행
            if tool_name == "general_faq":
                response = create_simple_response(slm, query, "faq_system")
                response = clean_markdown_formatting(response)
                return {
                    **state,
                    "messages": [result],
                    "response": response,
                    "ready_to_answer": True,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                    "intent_category": intent_category,
                }
            elif tool_name == "rag_search":
                # RAG 검색 실행
                from ..rag.vector_store import VectorStore
                vector_store = VectorStore()
                vector_store.get_index_ready()
                retrieved_docs = vector_store.similarity_search(query, k=3)
                response, sources = create_rag_response(slm, query, retrieved_docs)
                response = clean_markdown_formatting(response)
                return {
                    **state,
                    "messages": [result],
                    "response": response,
                    "sources": sources,
                    "retrieved_docs": retrieved_docs,
                    "context_text": format_context(retrieved_docs) if retrieved_docs else "",
                    "ready_to_answer": True,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                    "intent_category": intent_category,
                }
            elif tool_name == "product_extraction":
                # 상품명 추출 실행
                extracted_product = extract_product_name(slm, query)
                sub_category = classify_product_subcategory(extracted_product)
                return {
                    **state,
                    "messages": [result],
                    "product_name": extracted_product,
                    "product_extraction_result": {
                        "product_name": extracted_product,
                        "sub_category": sub_category,
                        "confidence": 0.9,
                        "reasoning": f"상품명 '{extracted_product}'을 추출하고 '{sub_category}'로 분류"
                    },
                    "ready_to_answer": False,  # product_search로 이어져야 함
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                    "intent_category": intent_category,
                }
            elif tool_name == "product_search":
                # 상품 검색 실행
                from ..rag.vector_store import VectorStore
                vector_store = VectorStore()
                vector_store.get_index_ready()
                product_name = state.get("product_name", "")
                
                if not product_name or product_name == "일반":
                    retrieved_docs = vector_store.similarity_search(query, k=3)
                else:
                    # 상품명으로 필터링 시도
                    filter_attempts = [
                        {"product_name": product_name},
                        {"keywords": {"$in": [product_name]}},
                        {"file_name": {"$regex": product_name, "$options": "i"}},
                        {"product_type": product_name}
                    ]
                    
                    retrieved_docs = []
                    for filter_dict in filter_attempts:
                        try:
                            retrieved_docs = vector_store.similarity_search(query, k=3, filter_dict=filter_dict)
                            if retrieved_docs:
                                break
                        except Exception:
                            continue
                    
                    if not retrieved_docs:
                        retrieved_docs = vector_store.similarity_search(query, k=3)
                
                response, sources = create_rag_response(slm, query, retrieved_docs)
                response = clean_markdown_formatting(response)
                return {
                    **state,
                    "messages": [result],
                    "response": response,
                    "sources": sources,
                    "retrieved_docs": retrieved_docs,
                    "context_text": format_context(retrieved_docs) if retrieved_docs else "",
                    "ready_to_answer": True,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                    "intent_category": intent_category,
                }
            elif tool_name == "answer":
                # 직접 답변
                return {
                    **state,
                    "messages": [result],
                    "ready_to_answer": True,
                    "n_tool_calling": state.get("n_tool_calling", 0) + 1,
                    "intent_category": intent_category,
                }
        
        return {
            **state,
            "messages": [result],
            "n_tool_calling": state.get("n_tool_calling", 0) + 1,
            "intent_category": intent_category,
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
    
    if state.get("ready_to_answer"):
        logger.info("[ROUTER] ready_to_answer=True - ANSWER로 라우팅")
        return "answer"
    
    # 첫 번째 턴인지 확인
    conversation_history = state.get("conversation_history", [])
    is_first_turn = not conversation_history or len(conversation_history) == 0
    
    logger.info(f"[ROUTER] conversation_history: {conversation_history}")
    logger.info(f"[ROUTER] is_first_turn: {is_first_turn}")
    
    # messages에서 마지막 메시지의 도구 호출 확인
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_name = last_message.tool_calls[0]['name']
            logger.info(f"[ROUTER] Selected tool: {tool_name}")
            return tool_name
    
    # 도구 호출이 없으면 기본적으로 RAG_SEARCH로
    logger.info("[ROUTER] No tool calls found - defaulting to RAG_SEARCH")
    return "rag_search"

def intent_classification_node(state: RAGState, slm: SLM = None) -> RAGState:
    """의도 분류 노드"""
    logger.info("🎯 [NODE] intent_classification_node 실행 시작")
    query = state.get("query", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
    try:
        # 의도 분류 프롬프트
        classification_prompt = f"""
            다음 질문을 정확히 분류해주세요: {query}

            분류 기준:
            1. general_banking_FAQs: 일반적인 은행 개념이나 금융 상식 질문
            - "예금이 뭐예요?", "적금이 뭐예요?", "대출이 뭐예요?", "카드가 뭐예요?" 등
            - 은행 업무의 기본 개념에 대한 질문
            - 특정 상품명이나 브랜드명이 언급되지 않은 일반적인 질문

            2. industry_policies_and_regulations: 은행업 규제 및 정책 관련
            - KYC, AML, 바젤3, 금융감독원 규정 등
            - 은행업계 전반의 규제나 정책에 대한 질문

            3. company_rules: KB금융그룹 내부 규칙 및 정책
            - 직원 복리후생, 휴가정책, 복장규정, 직원 교육, 인사정책 등
            - KB금융그룹 내부 정책이나 직원 관련 질문
            - "직원", "휴가", "복리후생", "인사", "교육" 등이 포함된 질문

            4. company_products: KB금융그룹의 구체적인 상품명이나 브랜드명이 언급된 경우
            - 햇살론, 닥터론, 로이어론, 내집마련디딤돌대출, 입주자 앞 대환대출, 버팀목 전세자금대출, 매직카대출, 군인 연금 협약 대출, 폐업지원 대환대출 등
            - "KB카드", "KB예금", "KB적금", "KB보험", "KB펀드" 등 KB 브랜드가 명시된 경우
            - "KB 햇살론", "햇살론", "KB카드" 등이 언급되면 무조건 company_products

            중요한 구분:
            - "예금이 뭐예요?" → general_banking_FAQs (일반적인 개념 질문)
            - "KB예금 상품에 대해 알려주세요" → company_products (구체적인 KB 상품)
            - "대출이 뭐예요?" → general_banking_FAQs (일반적인 개념 질문)
            - "햇살론 대출에 대해 알려주세요" → company_products (구체적인 상품명)
            - "직원 휴가 정책은 어떻게 되나요?" → company_rules (직원 관련 정책)
            - "복리후생 제도는 어떻게 되나요?" → company_rules (직원 관련 정책)

            예시:
            - "예금이 뭐예요?" → general_banking_FAQs
            - "적금이 뭐예요?" → general_banking_FAQs
            - "대출이 뭐예요?" → general_banking_FAQs
            - "햇살론 대출에 대해 알려주세요" → company_products
            - "KB 햇살론 대출 조건과 금리를 알려주세요" → company_products
            - "KB카드 혜택이 뭐예요?" → company_products
            - "오피스텔을 담보로 대출 신청을 원하는 고객에게 관련 상품 추천해주고, 정확한 신청 자격 알려줘" → company_products
            - "KYC 규정이 뭐예요?" → industry_policies_and_regulations
            - "직원 휴가 정책은 어떻게 되나요?" → company_rules
            - "바젤3 규정에 대해 알려주세요" → industry_policies_and_regulations

            질문: {query}

            분류 결과만 출력하세요 (general_banking_FAQs, industry_policies_and_regulations, company_rules, company_products 중 하나):
            """
        
        intent_category = slm.invoke(classification_prompt).strip()
        
        # 결과 정리
        intent_category = intent_category.lower().strip()
        
        # 키워드 기반 강화 분류
        query_lower = query.lower()
        logger.info(f"Original LLM classification: {intent_category}")
        logger.info(f"Query for keyword check: {query_lower}")
        
        if any(keyword in query_lower for keyword in ["직원", "휴가", "복리후생", "인사", "교육", "정책"]):
            intent_category = "company_rules"
            logger.info("Keyword-based classification: company_rules")
        elif any(keyword in query_lower for keyword in ["햇살론", "닥터론", "로이어론"]):
            intent_category = "company_products"
            logger.info("Keyword-based classification: company_products")
        elif any(keyword in query_lower for keyword in ["kyc", "aml", "바젤", "금융감독원", "규정"]):
            intent_category = "industry_policies_and_regulations"
            logger.info("Keyword-based classification: industry_policies_and_regulations")
        elif "company_products" in intent_category:
            intent_category = "company_products"
        elif "industry_policies_and_regulations" in intent_category:
            intent_category = "industry_policies_and_regulations"
        elif "company_rules" in intent_category:
            intent_category = "company_rules"
        else:
            intent_category = "general_banking_FAQs"
        
        logger.info(f"Intent classified as: {intent_category}")
        
        return {
            **state,
            "intent_category": intent_category
        }
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            **state,
            "intent_category": "general_banking_FAQs"  # 기본값
        }

def general_faq_node(state: RAGState, slm: SLM = None) -> RAGState:
    """일반 은행 FAQ 처리 노드 (SLM 직접 답변)"""
    logger.info("❓ [NODE] general_faq_node 실행 시작")
    query = state.get("query", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
    try:
        response = create_simple_response(slm, query, "faq_system")
        return {
            **state,
            "response": response,
            "ready_to_answer": True
        }
        
    except Exception as e:
        logger.error(f"General FAQ failed: {e}")
        return {
            **state,
            **create_error_response("faq_error")
        }

def product_extraction_node(state: RAGState, slm: SLM = None) -> RAGState:
    """상품명 추출 노드"""
    logger.info("🏷️ [NODE] product_extraction_node 실행 시작")
    query = state.get("query", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
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
    query = state.get("query", "")
    product_name = state.get("product_name", "")
    
    # SLM과 VectorStore 인스턴스 생성
    if slm is None:
        slm = SLM()
    vector_store = VectorStore()
    
    try:
        vector_store.get_index_ready()
        
        if not product_name or product_name == "일반":
            # 일반 검색
            retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K)
            logger.info(f"General search found {len(retrieved_docs)} documents")
        else:
            # 상품명으로 메타데이터 필터링 (여러 필드 시도)
            filter_attempts = [
                {"product_name": product_name},
                {"keywords": {"$in": [product_name]}},
                {"file_name": {"$regex": product_name, "$options": "i"}},
                {"product_type": product_name}
            ]
            
            retrieved_docs = []
            for filter_dict in filter_attempts:
                try:
                    retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K, filter_dict=filter_dict)
                    if retrieved_docs:
                        logger.info(f"Found {len(retrieved_docs)} documents using filter: {filter_dict}")
                        break
                except Exception as e:
                    logger.warning(f"Filter failed: {filter_dict}, error: {e}")
                    continue
            
            # 필터링된 결과가 없으면 일반 검색으로 폴백
            if not retrieved_docs:
                retrieved_docs = vector_store.similarity_search(query, k=DEFAULT_SEARCH_K)
                logger.info(f"No documents found for product '{product_name}', using general search")
                logger.info(f"General search found {len(retrieved_docs)} documents")
            else:
                logger.info(f"Found {len(retrieved_docs)} documents for product '{product_name}'")
        
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
    query = state.get("query", "")
    session_context = state.get("session_context")
    conversation_history = state.get("conversation_history", [])
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
    try:
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
            
    except Exception as e:
        logger.error(f"Session summary failed: {e}")
        # 폴백: 공통 함수 사용
        fallback_title = generate_session_title(query, slm)
        return {
            **state,
            "session_title": fallback_title,
            "ready_to_answer": False
        }

def rag_search_node(state: RAGState, slm: SLM = None, vector_store=None) -> RAGState:
    """RAG 검색 노드"""
    import time
    start_time = time.time()
    logger.info("[RAG_SEARCH] Starting document search")
    query = state.get("query", "")
    
    # SLM과 VectorStore 인스턴스 생성 (재사용 가능)
    if slm is None:
        slm = SLM()
    if vector_store is None:
        vector_store = VectorStore()
        vector_store.get_index_ready()  # 한 번만 초기화
    
    try:
        # 문서 검색 (성능 최적화를 위해 결과 수 제한)
        retrieved_docs = vector_store.similarity_search(query, k=3)# 더 적은 결과로 속도 향상
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
    response = state.get("response", "")
    
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    
    try:
        # 공통 함수를 사용하여 가드레일 검사
        compliant_response, violations = create_guardrail_response(slm, response)
        
        return {
            **state,
            "guardrail_decision": "COMPLIANT" if not violations else "VIOLATION",
            "violations": violations,
            "compliant_response": compliant_response,
            "response": compliant_response,
            "ready_to_answer": True
        }
        
    except Exception as e:
        logger.error(f"Guardrail check failed: {e}")
        return {
            **state,
            "guardrail_decision": "ERROR",
            "violations": ["가드레일 검사 오류"],
            "compliant_response": ERROR_MESSAGES["guardrail_error"],
            "response": ERROR_MESSAGES["guardrail_error"],
            "ready_to_answer": True
        }

def answer_node(state: RAGState) -> RAGState:
    """최종 답변 노드"""
    import time
    from datetime import datetime
    start_time = time.time()
    logger.info("[ANSWER] Preparing final response")
    response = state.get("response", "")
    
    if not response or not response.strip():
        # 기본 응답 생성 (RAG 검색이 실패한 경우에만)
        response = "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다. 다른 키워드로 다시 질문해주시거나 관련 부서에 문의해주세요."
        logger.info("[ANSWER] Using default response")
    
    # 마크다운 형식 제거
    response = clean_markdown_formatting(response)
    
    # 대화 턴 저장
    session_context = state.get("session_context")
    if session_context:
        from .session_manager import ConversationTurn
        turn = ConversationTurn(
            turn_id=state.get("turn_id", ""),
            timestamp=datetime.now(),
            user_query=state.get("query", ""),
            ai_response=response,
            category=state.get("intent_category", ""),
            product_name=state.get("product_name", ""),
            sources=state.get("sources", []),
            session_context={}  # 빈 딕셔너리로 초기화
        )
        session_manager.add_conversation_turn(session_context.session_id, turn)
        logger.info(f"✅ [NODE] 대화 턴 저장 완료: {session_context.session_id}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"✅ [NODE] answer_node 완료 - 실행시간: {execution_time:.2f}초")
    
    return {
        **state,
        "response": response,
        "ready_to_answer": True
    }

