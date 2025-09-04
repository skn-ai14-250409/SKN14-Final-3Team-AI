#!/usr/bin/env python3
"""
공통 유틸리티 함수들
- 서버 연결 확인
- 응답 시간 측정
- 에러 처리
- 로깅
"""
import requests
import time
import logging
from typing import Dict, Any, Optional, Tuple
from functools import wraps

# 로깅 설정
logger = logging.getLogger(__name__)

def check_server_health(base_url: str, timeout: int = 5) -> bool:
    """서버 상태 확인"""
    try:
        response = requests.get(f"{base_url}/healthcheck", timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"서버 상태 확인 실패: {e}")
        return False

def measure_response_time(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            response_time = end_time - start_time
            logger.info(f"{func.__name__} 실행 시간: {response_time:.3f}초")
            return result
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            logger.error(f"{func.__name__} 실행 실패 (시간: {response_time:.3f}초): {e}")
            raise
    return wrapper

def safe_api_call(func):
    """API 호출을 안전하게 처리하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            logger.error(f"{func.__name__}: 요청 시간 초과")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"{func.__name__}: 서버 연결 실패")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"{func.__name__}: API 요청 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"{func.__name__}: 예상치 못한 오류: {e}")
            raise
    return wrapper

def format_file_size(size_bytes: int) -> str:
    """파일 크기를 읽기 쉬운 형태로 변환"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """파일 확장자 검증"""
    if not filename:
        return False
    
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    return file_ext in allowed_extensions

def sanitize_filename(filename: str) -> str:
    """파일명 정리 (위험한 문자 제거)"""
    import re
    # 위험한 문자들을 안전한 문자로 대체
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 연속된 언더스코어를 하나로
    sanitized = re.sub(r'_+', '_', sanitized)
    # 앞뒤 공백 및 언더스코어 제거
    sanitized = sanitized.strip(' _')
    return sanitized

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """실패 시 재시도하는 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"{func.__name__} 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                        time.sleep(delay * (2 ** attempt))  # 지수 백오프
                    else:
                        logger.error(f"{func.__name__} 최종 실패: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """진행률 바 생성"""
    if total == 0:
        return "[" + " " * width + "] 0%"
    
    progress = int(width * current / total)
    bar = "[" + "█" * progress + " " * (width - progress) + "]"
    percentage = int(100 * current / total)
    return f"{bar} {percentage}%"

def parse_metadata_field(field_value: Any) -> Any:
    """메타데이터 필드 값 파싱 및 정리"""
    if isinstance(field_value, str):
        # 문자열 정리
        cleaned = field_value.strip()
        if cleaned.lower() in ['none', 'null', 'undefined', '']:
            return None
        return cleaned
    elif isinstance(field_value, list):
        # 리스트 정리
        cleaned_list = [item.strip() for item in field_value if item and str(item).strip()]
        return cleaned_list if cleaned_list else None
    elif isinstance(field_value, dict):
        # 딕셔너리 정리
        cleaned_dict = {}
        for key, value in field_value.items():
            cleaned_value = parse_metadata_field(value)
            if cleaned_value is not None:
                cleaned_dict[key] = cleaned_value
        return cleaned_dict if cleaned_dict else None
    
    return field_value

def extract_keywords_from_text(text: str, min_length: int = 2) -> list:
    """텍스트에서 한국어 키워드 추출"""
    import re
    
    if not text:
        return []
    
    # 한국어 단어 추출 (2글자 이상)
    korean_words = re.findall(r'[가-힣]{2,}', text)
    
    # 길이 필터링
    keywords = [word for word in korean_words if len(word) >= min_length]
    
    # 중복 제거 및 정렬
    return sorted(list(set(keywords)))

def calculate_text_similarity(text1: str, text2: str) -> float:
    """두 텍스트 간의 유사도 계산 (간단한 버전)"""
    if not text1 or not text2:
        return 0.0
    
    # 키워드 추출
    keywords1 = set(extract_keywords_from_text(text1))
    keywords2 = set(extract_keywords_from_text(text2))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # Jaccard 유사도
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

def format_timestamp(timestamp: Optional[float] = None) -> str:
    """타임스탬프를 읽기 쉬운 형태로 변환"""
    if timestamp is None:
        timestamp = time.time()
    
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def get_memory_usage() -> Dict[str, Any]:
    """메모리 사용량 정보 반환"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": format_file_size(memory_info.rss),  # 물리 메모리
            "vms": format_file_size(memory_info.vms),  # 가상 메모리
            "percent": process.memory_percent()  # 메모리 사용률
        }
    except ImportError:
        return {"error": "psutil 모듈이 설치되지 않음"}

def log_function_call(func_name: str, args: tuple = None, kwargs: dict = None):
    """함수 호출 로깅"""
    log_msg = f"함수 호출: {func_name}"
    
    if args:
        log_msg += f" | 위치 인자: {args}"
    if kwargs:
        log_msg += f" | 키워드 인자: {kwargs}"
    
    logger.info(log_msg)

def validate_environment_variables(required_vars: list) -> Tuple[bool, list]:
    """필수 환경변수 검증"""
    import os
    
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars
