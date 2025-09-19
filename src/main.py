from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.api.router import router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KB금융 RAG API",
    description="KB금융 문서 검색 및 질의응답 시스템",
    version="1.0.0"
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

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("KB금융 RAG API가 종료되었습니다.")