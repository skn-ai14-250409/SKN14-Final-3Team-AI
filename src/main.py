from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from src.api.router import router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # INFO 레벨로 변경 (더 많은 로그 표시)
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler("app.log", encoding="utf-8", mode="w")  # 파일 출력 (UTF8 인코딩)
    ],
    force=True  # 기존 설정 강제 덮어쓰기
)
logger = logging.getLogger(__name__)

# 모든 로거 레벨 설정
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("src.langgraph").setLevel(logging.INFO)
logging.getLogger("src.rag").setLevel(logging.INFO)
logging.getLogger("src.slm").setLevel(logging.INFO)
logging.getLogger("src.api").setLevel(logging.INFO)

# 특정 모듈 로깅 활성화
logging.getLogger("src.langgraph.nodes").setLevel(logging.INFO)
logging.getLogger("src.langgraph.agent").setLevel(logging.INFO)

app = FastAPI(
    title="KB금융 RAG API",
    description="KB금융 문서 검색 및 질의응답 시스템",
    version="1.0.0",
    default_response_class=JSONResponse
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],  # 실제 배포시에는 구체적인 도메인으로 변경
    allow_origins=["http://127.0.0.1:8000"],  # Django 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 포함
app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    """루트 엔드포인트 - API 상태 확인"""
    return {
        "message": "KB금융 RAG API가 정상적으로 작동 중입니다.",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "message": "API is running"}

# 애플리케이션 시작시 로그
@app.on_event("startup")
async def startup_event():
    logger.info("KB금융 RAG API가 시작되었습니다.")
    
    # 모델 사전 로딩 (선택사항)
    try:
        from src.langgraph.utils import get_shared_slm, get_shared_vector_store, get_memory_usage
        logger.info("모델 사전 로딩 중...")
        slm = get_shared_slm()
        vector_store = get_shared_vector_store()
        
        # 메모리 사용량 확인
        memory_info = get_memory_usage()
        logger.info(f"📊 [STARTUP] Memory usage: {memory_info['rss_mb']:.2f}MB")
        
        logger.info("✅ 모델 로딩 완료")
    except Exception as e:
        logger.warning(f"모델 사전 로딩 실패 (정상 동작): {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("KB금융 RAG API가 종료되었습니다.")