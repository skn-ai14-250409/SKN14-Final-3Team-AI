from typing import Literal, List, Optional, Dict
import logging
import anyio
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import os

from src.orchestrator import Orchestrator
from src.experimental.langgraph_rag import get_langgraph_workflow
from src.rag.vector_store import VectorStore
from src.config import VECTOR_STORE_INDEX_NAME, DATA_FOLDER_PATH
from src.constants import STATUS_SUCCESS, STATUS_FAIL

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ==============================================
# Pydantic 모델 정의 (Request/Response)
# ==============================================

# 기본 응답 모델
# RAG, Sub/Main 파이프라인, 데이터 관리 등 기본 응답 모델
class BaseResponse(BaseModel):
    status: Literal[STATUS_SUCCESS, STATUS_FAIL]

# 요청 모델들
# RAG 질의 요청 모델
class QueryRagInput(BaseModel):
    prompt: str

# 기존 워크플로우 실행 요청 모델
class LLMOnlyInput(BaseModel):
    prompt: str

# Intent 라우팅 기반 처리 요청 모델
class IntentRoutingInput(BaseModel):
    prompt: str

# 폴더 업로드 요청 모델
class IngestFolderReq(BaseModel):
    # ALLOWED_ROOT 기준 "상대 폴더 경로"만 받습니다. (예: "법률률", "branch_docs/여신")
    root_folder_path: str = Field(..., min_length=1, description="ALLOWED_ROOT-relative folder path")

# 벡터 삭제 요청 모델
class DeleteVectorsByConditionReq(BaseModel):
    field: str = Field(..., description="삭제할 메타데이터 필드명 (예: main_category, sub_category, file_name)")
    value: str = Field(..., description="삭제할 값 (예: 법률, 개인 신용대출, 특정파일.pdf)")

# Sub/Main 라우터 요청 모델
class IntentRouterInput(BaseModel):
    question: str

# 응답 모델들
# RAG 질의 응답 모델
class QueryRagResponse(BaseResponse):
    response: str
    sources: Optional[List[Dict]] = None  

# 기존 워크플로우 실행 응답 모델
class LLMOnlyResponse(BaseResponse):
    response: str

# Intent 라우팅 기반 처리 응답 모델
class IntentRoutingResponse(BaseResponse):
    response: str
    sources: Optional[List[Dict]] = []
    category: Optional[str] = ""

# 개별 파일 업로드 응답 모델
class UploadRagDocsResponse(BaseResponse):
    chunk_ids: List[str]
    message: str



# 벡터 삭제 조건 응답 모델
class DeleteVectorsByConditionResponse(BaseResponse):
    deleted_count: int
    message: str

# Sub/Main 라우터 응답 모델
class IntentRouterResponse(BaseResponse):
    route: Dict
    answer: str
    sources: List[Dict] = []

# ==============================================
# 상수 및 헬퍼 함수
# ==============================================

# 환경변수에서 데이터 폴더 경로를 가져오거나 기본값 사용
ALLOWED_ROOT = Path(DATA_FOLDER_PATH).expanduser().resolve()

# VectorStore 인스턴스를 반환하는 헬퍼 함수
def _get_vector_store() -> VectorStore:
    """VectorStore 인스턴스를 반환하는 헬퍼 함수"""
    return VectorStore()

# 경로 검증 및 우회 방지 헬퍼 함수
def _validate_and_resolve_target(relative_path_str: str) -> Path:
    """
    상대 경로를 검증하고 절대 경로로 변환합니다.
    
    Returns:
        Path: 검증된 절대 경로
        
    Raises:
        ValueError: 경로 검증 실패 시
        FileNotFoundError: 폴더가 존재하지 않을 때
        NotADirectoryError: 디렉터리가 아닐 때
    """
    rp = relative_path_str.strip()
    p = Path(rp)

    # 절대경로/드라이브/루트 시작 금지
    if p.is_absolute() or p.drive or rp.startswith(("/", "\\")):
        raise ValueError("absolute path is not allowed; provide a path relative to ALLOWED_ROOT")

    # 경로 우회 차단
    if ".." in p.parts:
        raise ValueError("path traversal ('..') is not allowed")

    target = (ALLOWED_ROOT / p).expanduser().resolve(strict=False)

    # ALLOWED_ROOT 하위 경로 확인 (startswith 대신 relative_to 사용)
    try:
        target.relative_to(ALLOWED_ROOT)
    except ValueError:
        raise ValueError("Forbidden path (outside of ALLOWED_ROOT)")

    if not target.exists():
        raise FileNotFoundError(f"folder not found: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {target}")

    return target


# ==============================================
# API 엔드포인트
# ==============================================

# 상태 확인
@router.get("/healthcheck")
async def healthcheck():
    return BaseResponse(status=STATUS_SUCCESS)

# RAG 질의
@router.post("/query_rag", response_model=QueryRagResponse)
async def query_rag(input: QueryRagInput):
    try:
        orchestrator = Orchestrator()
        result = orchestrator.query_rag(input.prompt)

        return QueryRagResponse(
            status=STATUS_SUCCESS,
            response=result["response"],
            sources=result.get("sources", [])  # 여기서 orchestrator가 반환하는 Document.metadata 리스트
        )
    except Exception as e:
        logger.error(f"query_rag failed: {e}")
        return QueryRagResponse(
            status=STATUS_FAIL,
            response=f"질의 처리 중 오류가 발생했습니다: {str(e)}",
            sources=[]
        )

# LLM 전용 답변 생성
@router.post("/answer_with_llm_only")
async def answer_with_llm_only(input: LLMOnlyInput):
    """LLM만 사용하는 간단한 답변 (RAG 없이)"""
    try:
        orchestrator = Orchestrator()
        response = orchestrator.answer_with_llm_only(input.prompt)
        return LLMOnlyResponse(
            status=STATUS_SUCCESS,
            response=response
        )
    except Exception as e:
        logger.error(f"answer_with_llm_only failed: {e}")
        return LLMOnlyResponse(
            status=STATUS_FAIL,
            response=f"LLM 답변 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/experimental/langgraph_rag")
async def experimental_langgraph_rag(input: QueryRagInput):
    """
    🧪 실험용 LangGraph RAG 엔드포인트
    
    기존 orchestrator와 동일한 기능을 LangGraph로 구현한 실험용 버전
    기존 코드는 그대로 유지하고 새로운 접근 방식을 테스트하기 위한 엔드포인트
    """
    logger.info(f"[EXPERIMENTAL] LangGraph RAG query: {input.prompt}")
    
    try:
        workflow = get_langgraph_workflow()
        result = workflow.run_workflow(input.prompt)
        
        return {
            "status": STATUS_SUCCESS,
            "response": result["response"],
            "sources": result["sources"],
            "category": result["category"],
            "experimental": True,
            "workflow_type": "langgraph"
        }
    except Exception as e:
        logger.error(f"[EXPERIMENTAL] LangGraph RAG failed: {e}")
        return {
            "status": STATUS_FAIL,
            "response": f"실험용 LangGraph RAG 처리 중 오류가 발생했습니다: {str(e)}",
            "sources": [],
            "category": "error",
            "experimental": True,
            "workflow_type": "langgraph"
        }

# Intent 라우팅 기반 처리
@router.post("/process_with_intent_routing")
async def process_with_intent_routing(input: IntentRoutingInput):
    """Intent 분류 후 적절한 처리 방식으로 답변 생성"""
    try:
        orchestrator = Orchestrator()
        result = orchestrator.process_with_intent_routing(input.prompt)
        
        # result가 딕셔너리인 경우 (새로운 형식)
        if isinstance(result, dict):
            return IntentRoutingResponse(
                status=STATUS_SUCCESS,
                response=result.get("response", ""),
                sources=result.get("sources", []),
                category=result.get("category", "")
            )
        # result가 문자열인 경우 (기존 형식)
        else:
            return IntentRoutingResponse(
                status=STATUS_SUCCESS,
                response=result,
                sources=[],
                category=""
            )
    except Exception as e:
        logger.error(f"process_with_intent_routing failed: {e}")
        return IntentRoutingResponse(
            status=STATUS_FAIL,
            response=f"Intent 라우팅 처리 중 오류가 발생했습니다: {str(e)}",
            sources=[],
            category=""
        )


# 모든 벡터 삭제
@router.delete("/delete_all_vectors")
async def delete_all_vectors():
    try:
        orchestrator = Orchestrator()
        orchestrator.delete_all_vectors()
        return BaseResponse(status=STATUS_SUCCESS)
    except Exception as e:
        logger.error(f"delete_all_vectors failed: {e}")
        raise HTTPException(status_code=500, detail=f"벡터 삭제 중 오류가 발생했습니다: {str(e)}")

# 조건부 벡터 삭제
@router.delete("/delete_vectors_by_condition")
async def delete_vectors_by_condition(req: DeleteVectorsByConditionReq):
    """특정 조건에 맞는 벡터만 삭제"""
    try:
        orchestrator = Orchestrator()
        deleted_count = orchestrator.delete_vectors_by_condition(req.field, req.value)
        
        return DeleteVectorsByConditionResponse(
            status=STATUS_SUCCESS,
            deleted_count=deleted_count,
            message=f"'{req.field}' = '{req.value}' 조건에 맞는 {deleted_count}개 벡터가 삭제되었습니다."
        )
    except Exception as e:
        logger.error(f"delete_vectors_by_condition failed: {e}")
        raise HTTPException(status_code=500, detail=f"조건부 벡터 삭제 중 오류가 발생했습니다: {str(e)}")

# 개별 파일 업로드
@router.post("/upload_docs_to_rag")
async def upload_docs_to_rag(files: List[UploadFile] = File(...)):
    try:
        orchestrator = Orchestrator()
        all_chunk_ids: List[str] = []
        for file in files:
            chunk_ids = await orchestrator.upload_docs_to_rag(file)
            all_chunk_ids += chunk_ids
        return UploadRagDocsResponse(
            status=STATUS_SUCCESS,
            chunk_ids=all_chunk_ids,
            message=f"{len(files)}개 파일이 성공적으로 업로드되었습니다."
        )
    except Exception as e:
        logger.error(f"upload_docs_to_rag failed: {e}")
        raise HTTPException(status_code=500, detail=f"파일 업로드 중 오류가 발생했습니다: {str(e)}")

# 폴더 업로드
@router.post("/ingest_folder")
async def ingest_folder(req: IngestFolderReq):
    """
    JSON Body 예:
    { "root_folder_path": "강령" }
    """
    logger.info(f"Starting folder ingestion for: {req.root_folder_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"ALLOWED_ROOT: {ALLOWED_ROOT}")
    
    try:
        target = _validate_and_resolve_target(req.root_folder_path)
    except ValueError as e:
        logger.error(f"Path validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Folder not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        logger.error(f"Not a directory: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    orchestrator = Orchestrator()
    try:
        uploaded_ids = await anyio.to_thread.run_sync(
            orchestrator.upload_folder_to_rag,
            str(target),
        )
        logger.info(f"Successfully uploaded {len(uploaded_ids)} chunks")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")

    return {
        "status": STATUS_SUCCESS,
        "allowed_root": str(ALLOWED_ROOT),
        "resolved_target": str(target),
        "uploaded_chunk_ids_count": len(uploaded_ids),
    }

# 벡터 스토어 초기화

# 상태 확인
@router.get("/vector_store_stats")
async def get_vector_store_stats():
    """벡터 스토어 통계 정보 조회"""
    try:
        store = _get_vector_store()
        stats = store.get_index_stats()
        return {
            "status": STATUS_SUCCESS,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# 카테고리별 RAG 질의
@router.post("/query_rag_by_category")
async def query_rag_by_category(input: QueryRagInput, category: str):
    """카테고리별 RAG 질의"""
    try:
        store = _get_vector_store()
        results = store.similarity_search_by_category(input.prompt, category)
        
        # 결과를 orchestrator 형식으로 변환
        sources = []
        for doc in results:
            meta = dict(doc.metadata or {})
            sources.append(meta)
        
        return QueryRagResponse(
            status=STATUS_SUCCESS, 
            response=f"Found {len(results)} results in category '{category}'",
            sources=sources
        )
    except Exception as e:
        logger.error(f"Category search failed: {e}")
        return QueryRagResponse(
            status=STATUS_FAIL,
            response=f"카테고리별 검색 중 오류가 발생했습니다: {str(e)}",
            sources=[]
        )

# 폴더별 RAG 질의
@router.post("/query_rag_by_folder")
async def query_rag_by_folder(input: QueryRagInput, main_category: str, sub_category: str = None):
    """폴더별 RAG 질의 (강령, 법률, 상품 등)"""
    try:
        store = _get_vector_store()
        results = store.similarity_search_by_folder(input.prompt, main_category, sub_category)
        
        # 결과를 orchestrator 형식으로 변환
        sources = []
        for doc in results:
            meta = dict(doc.metadata or {})
            sources.append(meta)
        
        folder_path = f"{main_category}"
        if sub_category:
            folder_path += f"/{sub_category}"
        
        return QueryRagResponse(
            status=STATUS_SUCCESS,
            response=f"Found {len(results)} results in folder '{folder_path}'",
            sources=sources
        )
    except Exception as e:
        logger.error(f"Folder search failed: {e}")
        return QueryRagResponse(
            status=STATUS_FAIL,
            response=f"폴더별 검색 중 오류가 발생했습니다: {str(e)}",
            sources=[]
        )