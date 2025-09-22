"""
LangGraph 공통 유틸리티 함수들
============================
중복 코드를 제거하고 공통 기능을 제공하는 유틸리티 모듈
"""

import json
import os
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
import yaml

logger = logging.getLogger(__name__)

# ========== 상수 정의 ==========
DEFAULT_SEARCH_K = 5
DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_MESSAGES = 100

# ========== 에러 메시지 ==========
ERROR_MESSAGES = {
    "general_error": "죄송합니다. 처리 중 오류가 발생했습니다.",
    "search_error": "죄송합니다. 검색 중 오류가 발생했습니다. 다시 시도해주세요.",
    "no_documents": "죄송합니다. 관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해보시겠어요?",
    "guardrail_error": "죄송합니다. 응답 검증 중 오류가 발생했습니다.",
    "faq_error": "죄송합니다. 일반적인 은행 FAQ 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
    "product_error": "죄송합니다. 상품 검색 중 오류가 발생했습니다. 다시 시도해주세요.",
    "extraction_error": "죄송합니다. 상품명 추출 중 오류가 발생했습니다.",
    "chitchat_error": "안녕하세요! KB금융그룹 상담을 도와드리겠습니다."
}

# ========== 프롬프트 관리 함수들 ==========

def load_prompts() -> Dict[str, Any]:
    """Load prompts from YAML file"""
    current_dir = os.path.dirname(__file__)
    prompts_path = os.path.join(current_dir, "prompts.yaml")
    
    try:
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Prompts file not found: {prompts_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading prompts: {e}")
        return {}


def get_prompt(category: str, **kwargs) -> str:
    """
    Get formatted prompt from YAML template
    
    Args:
        category: Prompt category (supervisor, chitchat, etc.)
        **kwargs: Variables to format into the prompt template
        
    Returns:
        Formatted prompt string
    """
    prompts = load_prompts()
    
    if category not in prompts.get("system_prompts", {}):
        logger.warning(f"Prompt category '{category}' not found")
        return f"프롬프트를 찾을 수 없습니다: {category}"
    
    prompt_template = prompts["system_prompts"][category]["system"]
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Prompt formatting error: {e}")
        return f"프롬프트 변수 오류: {e}"


def get_error_message(message_key: str, **kwargs) -> str:
    """
    Get error message with formatting
    
    Args:
        message_key: Key in error_messages section
        **kwargs: Variables to format into the message
        
    Returns:
        Formatted error message
    """
    prompts = load_prompts()
    
    if "error_messages" not in prompts:
        return "기본 에러 메시지를 불러올 수 없습니다."
    
    error_messages = prompts["error_messages"]
    
    if message_key not in error_messages:
        return "해당 에러 메시지를 찾을 수 없습니다."
    
    message_template = error_messages[message_key]
    try:
        return message_template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Error message formatting error: {e}")
        return f"에러 메시지 변수 오류: {e}"


# ========== 기존 프롬프트 템플릿 (호환성 유지) ==========
SYSTEM_PROMPTS = {
    "rag_system": """당신은 KB금융그룹의 전문 상담사입니다.

        다음 문서들을 참고하여 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요:

        {context_text}

        지침:
        1. 문서 내용을 바탕으로 정확한 정보를 제공하세요
        2. 문서에 없는 정보는 추측하지 마세요
        3. 친근하고 전문적인 톤으로 답변하세요
        4. 문서 출처는 언급하지 마세요 (링크나 참조 형태로 표시하지 마세요)
        5. 답변은 5줄 이내로 간결하게 작성하세요""",
            
    "faq_system": """당신은 KB금융그룹의 전문 상담사입니다.
        사용자의 일반적인 은행 FAQ 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요.

        사용자 질문: {query}

        답변 지침:
        1. 일반적인 은행 업무에 대한 정확한 정보를 제공하세요
        2. 예금, 적금, 대출, 카드 등 기본적인 금융 상품에 대해 설명하세요
        3. 친근하고 이해하기 쉬운 언어로 답변하세요
        4. 답변은 3-5줄로 간결하게 작성하세요

        KB금융그룹의 전문 상담사로서 신뢰할 수 있는 정보를 제공해주세요.""",
        
    "chitchat_system": """당신은 친근하고 도움이 되는 KB금융그룹 상담사입니다.
        사용자의 인사, 감사, 일반적인 대화에 자연스럽게 응답해주세요.

        사용자 메시지: {query}

        친근하고 전문적인 톤으로 응답해주세요.""",
        
        "guardrail_system": """다음 응답이 KB금융그룹의 가드레일을 준수하는지 검사하세요.

        응답: {response}

        검사 항목:
        1. 금융 상품 추천이나 투자 조언이 있는가?
        2. 개인정보나 민감한 정보가 노출되었는가?
        3. 법적 조언이나 세무 조언이 있는가?
        4. 부적절한 내용이나 비윤리적인 내용이 있는가?

        준수 여부: [COMPLIANT/VIOLATION]
        위반 사항: (위반이 있는 경우에만)""",
        
    "product_extraction_system": """다음 질문에서 KB금융그룹의 상품명을 추출하세요.

        질문: {query}

        추출할 상품명 예시:
        - 햇살론, 내집마련디딤돌대출, 입주자 앞 대환대출, 버팀목 전세자금대출, 매직카대출, 군인 연금 협약 대출, 폐업지원 대환대출, 닥터론, 로이어론
        - 대출상품, 예금상품, 적금상품, 보험상품, 펀드상품

        상품명이 있으면 정확히 추출하고, 없으면 "일반"이라고 답하세요.
        답변 형식: [상품명]""",
        
        "supervisor_system": """당신은 KB금융그룹의 중앙 관리자입니다.
        사용자의 요청을 분석하고 적절한 도구를 선택해주세요.

        사용자 요청: {query}
        첫 대화 여부: {is_first_turn}
        의도 분류: {intent_category}
        현재 응답 상태: {response_status}
        추출된 상품명: {product_name}

        의도 분류별 권장 도구:
        - general_banking_FAQs: 일반적인 은행 FAQ → general_faq (SLM 직접 답변)
        - industry_policies_and_regulations: 규제/정책 관련 → rag_search  
        - company_rules: 회사 내부 규칙 → rag_search
        - company_products: 회사 상품 관련 → product_extraction (상품명 있으면) 또는 rag_search

        상황에 맞는 도구를 선택하세요:
        - chitchat: 인사, 감사, 일반 대화
        - general_faq: 일반적인 은행 FAQ (예금, 적금, 대출 기본 개념 등)
        - rag_search: 문서 검색이 필요한 구체적인 질문
        - product_extraction: 특정 상품명이 언급된 질문에서 상품명 추출
        - product_search: 추출된 상품명으로 상품 정보 검색
        - session_summary: 첫 대화일 때 세션 요약 생성
        - guardrail_check: 응답이 생성된 후 가드레일 검사 필요 (응답이 있을 때만)
        - answer: 충분한 정보로 최종 답변 준비됨

        {response_guidance}
        {product_guidance}

        신중하게 분석하고 가장 적절한 도구를 선택해주세요."""
    }

def create_title_generation_prompt(query: str) -> str:
    """
    제목 생성 프롬프트 생성
    
    Args:
        query (str): 사용자 쿼리
        
    Returns:
        str: 제목 생성을 위한 프롬프트
    """
    return f"""당신은 '챗봇 세션 제목 생성기'입니다. 사용자의 질문을 보고 
        이 세션을 대표할 매우 간결한 한 줄 제목을 생성하세요.

        사용자 질문: {query}

        출력: 제목 텍스트만 한 줄로 출력
        - 한국어로 간결하게 (반드시 15자 이하)
        - 불필요한 설명 없이 제목만 반환
        - 예시: '햇살론 문의', '대출 문의', '상품 안내'
        - 중요: 15자를 초과하면 안됩니다!"""


def truncate_title(title: str, max_length: int = 15) -> str:
    """
    제목을 지정된 길이로 자르기
    
    Args:
        title (str): 원본 제목
        max_length (int): 최대 길이 (기본값: 15)
        
    Returns:
        str: 잘린 제목
    """
    if len(title) > max_length:
        return title[:max_length].rstrip()
    return title


def generate_session_title(query: str, slm_instance) -> str:
    """
    세션 제목 생성 (완전한 로직)
    
    Args:
        query (str): 사용자 쿼리
        slm_instance: SLM 인스턴스
        
    Returns:
        str: 생성된 세션 제목
    """
    try:
        # LLM으로 제목 생성
        title_prompt = create_title_generation_prompt(query)
        logger.info(f"Title generation prompt: {title_prompt}")
        
        raw_title = (slm_instance.invoke(title_prompt) or "").strip()
        logger.info(f"Raw title from LLM: '{raw_title}'")
        
        if raw_title:
            session_title = raw_title.splitlines()[0].strip()
            # 15자 제한 강제 적용
            session_title = truncate_title(session_title, 15)
            logger.info(f"Final session title: '{session_title}'")
            return session_title
        else:
            # LLM 실패시 폴백
            fallback_title = truncate_title(query, 15)
            logger.info(f"LLM failed, using fallback title: '{fallback_title}'")
            return fallback_title
            
    except Exception as e:
        # 에러 발생시 폴백
        fallback_title = truncate_title(query, 15)
        logger.error(f"Title generation error: {e}, using fallback: '{fallback_title}'")
        return fallback_title


def format_context(documents: List[Document]) -> str:
    """
    문서들을 컨텍스트 형식으로 포맷팅 (성능 최적화)
    
    Args:
        documents: Document 리스트
        
    Returns:
        str: 포맷팅된 컨텍스트 문자열
    """
    lines = []
    max_context_length = 2000  # 컨텍스트 길이 제한
    
    for i, doc in enumerate(documents, 1):
        src = doc.metadata.get("source", f"document_{i}") if doc.metadata else f"document_{i}"
        snippet = doc.page_content.strip()
        if not snippet:
            continue
        
        # 각 문서의 내용을 500자로 제한
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
        
        lines.append(f"[source: {src}]\n{snippet}")
        
        # 전체 컨텍스트 길이 제한
        current_length = len("\n---\n".join(lines))
        if current_length > max_context_length:
            break
    
    return "\n---\n".join(lines)


def extract_sources_from_docs(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    문서들에서 소스 정보 추출
    
    Args:
        documents: Document 리스트
        
    Returns:
        List[Dict]: 소스 정보 리스트
    """
    sources = []
    for doc in documents:
        metadata = doc.metadata or {}
        source_info = dict(metadata)
        source_info['text'] = doc.page_content
        sources.append(source_info)
    return sources


def create_rag_response(slm_instance, query: str, documents: List[Document]) -> tuple[str, List[Dict[str, Any]]]:
    """
    RAG 검색 결과로부터 응답 생성
    
    Args:
        slm_instance: SLM 인스턴스
        query: 사용자 쿼리
        documents: 검색된 문서들
        
    Returns:
        tuple: (응답 텍스트, 소스 정보 리스트)
    """
    if not documents:
        return ERROR_MESSAGES["no_documents"], []
    
    # 컨텍스트 생성
    context_text = format_context(documents)
    
    # LLM으로 응답 생성
    system_prompt = SYSTEM_PROMPTS["rag_system"].format(context_text=context_text)
    messages = [HumanMessage(content=system_prompt), HumanMessage(content=query)]
    response = slm_instance.invoke(messages)
    
    # 소스 정보 추출
    sources = extract_sources_from_docs(documents)
    
    return response, sources


def create_simple_response(slm_instance, query: str, prompt_type: str) -> str:
    """
    간단한 프롬프트로 응답 생성
    
    Args:
        slm_instance: SLM 인스턴스
        query: 사용자 쿼리
        prompt_type: 프롬프트 타입 (faq_system, chitchat_system 등)
        
    Returns:
        str: 생성된 응답
    """
    try:
        prompt = SYSTEM_PROMPTS[prompt_type].format(query=query)
        return slm_instance.invoke(prompt)
    except Exception:
        return ERROR_MESSAGES.get(f"{prompt_type.replace('_system', '_error')}", ERROR_MESSAGES["general_error"])


def create_guardrail_response(slm_instance, response: str) -> tuple[str, List[str]]:
    """
    가드레일 검사 및 응답 생성
    
    Args:
        slm_instance: SLM 인스턴스
        response: 검사할 응답
        
    Returns:
        tuple: (준수 응답, 위반 사항 리스트)
    """
    try:
        prompt = SYSTEM_PROMPTS["guardrail_system"].format(response=response)
        guardrail_result = slm_instance.invoke(prompt).strip()
        
        if "VIOLATION" in guardrail_result.upper():
            # 위반이 있는 경우 안전한 응답으로 대체
            compliant_response = "죄송합니다. 해당 질문에 대해서는 정확한 답변을 드리기 어렵습니다. KB금융그룹 고객센터(1588-9999)로 문의해주시기 바랍니다."
            violations = ["가드레일 위반 감지"]
        else:
            compliant_response = response
            violations = []
        
        return compliant_response, violations
        
    except Exception:
        return ERROR_MESSAGES["guardrail_error"], ["가드레일 검사 오류"]


def extract_product_name(slm_instance, query: str) -> str:
    """
    상품명 추출 (새로운 프롬프트 시스템 사용)
    
    Args:
        slm_instance: SLM 인스턴스
        query: 사용자 쿼리
        
    Returns:
        str: 추출된 상품명
    """
    try:
        # 새로운 프롬프트 시스템 사용
        prompt = get_prompt("product_extraction", query=query)
        extracted_product = slm_instance.invoke(prompt).strip()
        
        # 대괄호 제거 및 정리
        if extracted_product.startswith('[') and extracted_product.endswith(']'):
            extracted_product = extracted_product[1:-1].strip()
        
        return extracted_product
        
    except Exception as e:
        logger.error(f"Product extraction failed: {e}")
        return "일반"


def classify_product_subcategory(product_name: str) -> str:
    """
    상품명을 기반으로 서브 카테고리 분류
    
    Args:
        product_name: 추출된 상품명
        
    Returns:
        str: 서브 카테고리
    """
    from .models import LoanSubType, DepositSubType, CardSubType, SavingsSubType
    
    product_lower = product_name.lower()
    
    # 대출 상품 분류
    if any(keyword in product_lower for keyword in ["햇살론", "닥터론", "로이어론"]):
        return LoanSubType.PERSONAL_CREDIT.value
    elif any(keyword in product_lower for keyword in ["내집마련", "입주자", "대환"]):
        return LoanSubType.PERSONAL_HOUSING_FUND.value
    elif any(keyword in product_lower for keyword in ["버팀목", "전세"]):
        return LoanSubType.PERSONAL_SECURED_JEONSE.value
    elif any(keyword in product_lower for keyword in ["매직카", "자동차"]):
        return LoanSubType.PERSONAL_AUTO.value
    elif any(keyword in product_lower for keyword in ["군인", "폐업", "기업"]):
        return LoanSubType.BUSINESS_LOAN.value
    
    # 예금/적금 상품 분류
    elif any(keyword in product_lower for keyword in ["예금", "정기예금"]):
        return DepositSubType.TIME_DEPOSIT.value
    elif any(keyword in product_lower for keyword in ["적금", "정기적금"]):
        return SavingsSubType.REGULAR_SAVINGS.value
    elif any(keyword in product_lower for keyword in ["자유적금"]):
        return SavingsSubType.FREE_SAVINGS.value
    elif any(keyword in product_lower for keyword in ["주택청약"]):
        return SavingsSubType.HOUSING_SAVINGS.value
    
    # 카드 상품 분류
    elif any(keyword in product_lower for keyword in ["카드", "신용카드"]):
        return CardSubType.CREDIT_CARD.value
    elif any(keyword in product_lower for keyword in ["체크카드"]):
        return CardSubType.DEBIT_CARD.value
    
    # 기본값
    else:
        return "일반"


def create_supervisor_prompt(query: str, is_first_turn: bool, intent_category: str, 
                           has_response: bool, extracted_product: str, has_product_name: bool) -> str:
    """
    슈퍼바이저 프롬프트 생성
    
    Args:
        query: 사용자 쿼리
        is_first_turn: 첫 대화 여부
        intent_category: 의도 분류
        has_response: 응답 생성 여부
        extracted_product: 추출된 상품명
        has_product_name: 상품명 존재 여부
        
    Returns:
        str: 슈퍼바이저 프롬프트
    """
    response_status = "응답 생성됨" if has_response else "응답 없음"
    product_name = extracted_product if has_product_name else "없음"
    
    response_guidance = "응답이 이미 생성되었으므로 guardrail_check를 선택하세요." if has_response else ""
    product_guidance = "상품명이 추출되었으므로 반드시 product_search를 선택하세요." if has_product_name and not has_response else ""
    
    try:
        return get_prompt("supervisor", 
            query=query,
            is_first_turn=is_first_turn,
            intent_category=intent_category,
            response_status=response_status,
            product_name=product_name,
            response_guidance=response_guidance,
            product_guidance=product_guidance
        )
    except KeyError as e:
        logger.error(f"Prompt formatting error: {e}")
        # 폴백 프롬프트 사용
        return f"""당신은 KB금융그룹의 중앙 관리자입니다. 사용자의 질문을 분석하여 적절한 도구를 선택하세요.
        
        사용자 질문: {query}
        첫 대화 여부: {is_first_turn}
        의도 분류: {intent_category}
        
        **중요**: 첫 대화(is_first_turn=True)일 때는 반드시 session_summary를 먼저 선택하세요.
        
        적절한 도구를 선택하고 reasoning을 제공하세요."""

def create_error_response(error_type: str, **kwargs) -> Dict[str, Any]:
    """
    에러 응답 생성
    
    Args:
        error_type: 에러 타입
        **kwargs: 추가 파라미터
        
    Returns:
        Dict: 에러 응답 딕셔너리
    """
    return {
        "response": ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["general_error"]),
        "sources": kwargs.get("sources", []),
        "ready_to_answer": True
    }
