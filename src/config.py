import os
from dotenv import load_dotenv
import logging

# 환경변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# 필수 환경변수 검증
def get_required_env(key: str, default: str = None) -> str:
    """필수 환경변수를 가져오고 없으면 기본값을 사용합니다."""
    value = os.environ.get(key, default)
    if not value:
        logger.warning(f"환경변수 {key}가 설정되지 않았습니다.")
    return value

def get_required_int_env(key: str, default: int) -> int:
    """정수형 환경변수를 가져오고 없으면 기본값을 사용합니다."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning(f"환경변수 {key}가 올바른 정수가 아닙니다. 기본값 {default}을 사용합니다.")
        return default

# AI 모델 설정
MODEL_KEY = get_required_env("MODEL_KEY")
MODEL_NAME = get_required_env("MODEL_NAME", "gpt-3.5-turbo")

# Pinecone 설정
PINECONE_KEY = get_required_env("PINECONE_KEY")

# 임베딩 설정
EMBEDDING_BACKEND = get_required_env("EMBEDDING_BACKEND")  # openai 또는 huggingface
EMBEDDING_MODEL_NAME = get_required_env("EMBEDDING_MODEL_NAME") # 예: "text-embedding-3-small" 또는 "bge-m3-small"

# 벡터 스토어 설정
VECTOR_STORE_INDEX_NAME = get_required_env("VECTOR_STORE_INDEX_NAME")
PINECONE_METRIC = get_required_env("PINECONE_METRIC")

# 문서 처리 설정
CHUNK_SIZE = get_required_int_env("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = get_required_int_env("CHUNK_OVERLAP", 200)

# 데이터 폴더 경로
DATA_FOLDER_PATH = get_required_env("DATA_FOLDER_PATH", "../SKN14-Final-3Team-Data")

# 설정 검증
def validate_config():
    """설정값들을 검증합니다."""
    global EMBEDDING_BACKEND
    
    missing_configs = []
    
    if not MODEL_KEY:
        missing_configs.append("MODEL_KEY")
    if not PINECONE_KEY:
        missing_configs.append("PINECONE_KEY")
    
    if missing_configs:
        logger.error(f"필수 환경변수가 누락되었습니다: {', '.join(missing_configs)}")
        raise ValueError(f"필수 환경변수가 누락되었습니다: {', '.join(missing_configs)}")
    
    if EMBEDDING_BACKEND not in ["openai", "huggingface"]:
        logger.warning(f"알 수 없는 임베딩 백엔드: {EMBEDDING_BACKEND}. 'openai'로 설정합니다.")
        EMBEDDING_BACKEND = "openai"
    
    logger.info("설정 검증이 완료되었습니다.")

# 초기화시 설정 검증
validate_config()

