from typing import Literal, List, Optional, Dict
import logging
import anyio
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pathlib import Path
import os

from src.orchestrator import Orchestrator
from src.rag.vector_store import VectorStore
from src.rag.document_loader import DocumentLoader
from src.config import VECTOR_STORE_INDEX_NAME

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class BaseResponse(BaseModel):
    status: Literal["successful", "fail"]

class QueryRagInput(BaseModel):
    prompt: str

class QueryRagResponse(BaseResponse):
    response: str
    sources: Optional[List[Dict]] = None  

class RunWorkflowInput(BaseModel):
    prompt: str

class RunWorkflowResponse(BaseResponse):
    response: str

class UploadRagDocsResponse(BaseResponse):
    chunk_ids: List[str]
    message: str

class InitializeVectorStoreResponse(BaseResponse):
    message: str
    documents_count: int
    chunk_ids: List[str]

class VectorStoreStatusResponse(BaseResponse):
    index_exists: bool
    index_name: str
    message: str

class IngestFolderReq(BaseModel):
    # ALLOWED_ROOT 기준 "상대 폴더 경로"만 받습니다. (예: "강릉", "branch_docs/여신")
    root_folder_path: str = Field(..., min_length=1, description="ALLOWED_ROOT-relative folder path")


# 환경변수에서 데이터 폴더 경로를 가져오거나 기본값 사용
DATA_FOLDER_PATH = os.getenv("DATA_FOLDER_PATH", "SKN14-Final-3Team-Data")
ALLOWED_ROOT = Path(DATA_FOLDER_PATH).expanduser().resolve()

def _validate_and_resolve_target(relative_path_str: str) -> Path:
    rp = relative_path_str.strip()
    p = Path(rp)

    # 절대경로/드라이브/루트 시작 금지
    if p.is_absolute() or p.drive or rp.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="absolute path is not allowed; provide a path relative to ALLOWED_ROOT")

    # 경로 우회 차단
    if ".." in p.parts:
        raise HTTPException(status_code=400, detail="path traversal ('..') is not allowed")

    target = (ALLOWED_ROOT / p).expanduser().resolve(strict=False)

    # ALLOWED_ROOT 하위 경로 확인 (startswith 대신 relative_to 사용)
    try:
        target.relative_to(ALLOWED_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden path (outside of ALLOWED_ROOT)")

    if not target.exists():
        raise HTTPException(status_code=404, detail=f"folder not found: {target}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    return target





@router.get("/healthcheck")
async def healthcheck():
    return BaseResponse(status="successful")

@router.post("/query_rag", response_model=QueryRagResponse)
async def query_rag(input: QueryRagInput):
    orchestrator = Orchestrator()
    result = orchestrator.query_rag(input.prompt)

    return QueryRagResponse(
        status="successful",
        response=result["response"],
        sources=result.get("sources", [])  # 여기서 orchestrator가 반환하는 Document.metadata 리스트
    )

@router.delete("/delete_all_vectors")
async def delete_all_vectors():
    orchestrator = Orchestrator()
    orchestrator.delete_all_vectors()
    return BaseResponse(status="successful")

class DeleteVectorsByConditionReq(BaseModel):
    field: str = Field(..., description="삭제할 메타데이터 필드명")
    value: str = Field(..., description="삭제할 값")

class DeleteVectorsByConditionResponse(BaseResponse):
    deleted_count: int
    message: str

@router.delete("/delete_vectors_by_condition")
async def delete_vectors_by_condition(req: DeleteVectorsByConditionReq):
    """특정 조건에 맞는 벡터만 삭제"""
    orchestrator = Orchestrator()
    deleted_count = orchestrator.delete_vectors_by_condition(req.field, req.value)
    
    return DeleteVectorsByConditionResponse(
        status="successful",
        deleted_count=deleted_count,
        message=f"'{req.field}' = '{req.value}' 조건에 맞는 {deleted_count}개 벡터가 삭제되었습니다."
    )

@router.post("/run_worflow")
async def run_worflow(input: RunWorkflowInput):
    orchestrator = Orchestrator()
    response = orchestrator.run_workflow(input.prompt)
    return RunWorkflowResponse(
        status="successful",
        response=response
    )

@router.post("/upload_docs_to_rag")
async def upload_docs_to_rag(files: List[UploadFile] = File(...)):
    orchestrator = Orchestrator()
    all_chunk_ids: List[str] = []
    for file in files:
        chunk_ids = await orchestrator.upload_docs_to_rag(file)
        all_chunk_ids += chunk_ids
    return UploadRagDocsResponse(
        status="successful",
        chunk_ids=all_chunk_ids
    )


@router.post("/ingest_folder")
async def ingest_folder(req: IngestFolderReq):
    """
    JSON Body 예:
    { "root_folder_path": "강령" }
    """
    logger.info(f"Starting folder ingestion for: {req.root_folder_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"ALLOWED_ROOT: {ALLOWED_ROOT}")
    
    target = _validate_and_resolve_target(req.root_folder_path)

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
        "status": "ingest_finished",
        "allowed_root": str(ALLOWED_ROOT),
        "resolved_target": str(target),
        "uploaded_chunk_ids_count": len(uploaded_ids),
    }

@router.post("/initialize_vector_store", response_model=InitializeVectorStoreResponse)
async def initialize_vector_store(background_tasks: BackgroundTasks):
    """
    벡터 스토어를 초기화하고 전체 데이터 폴더의 문서들을 업로드합니다.
    이는 기존에 vector_store.py를 직접 실행하던 것을 API로 대체합니다.
    """
    try:
        logger.info(f"Starting vector store initialization with data from: {ALLOWED_ROOT}")
        
        # 백그라운드에서 실행할 함수 정의
        def initialize_store():
            try:
                # 1. DocumentLoader로 문서 로드
                docs = DocumentLoader.process_folder_and_get_chunks(str(ALLOWED_ROOT))
                logger.info(f"Loaded {len(docs)} documents from {ALLOWED_ROOT}")
                
                # 2. VectorStore 초기화
                store = VectorStore(index_name=VECTOR_STORE_INDEX_NAME)
                store.get_index_ready()
                
                # 3. Pinecone에 문서 추가
                ids = store.add_documents_to_index(docs)
                logger.info(f"Added {len(ids)} documents to Pinecone index")
                
                return len(docs), ids
            except Exception as e:
                logger.error(f"Vector store initialization failed: {e}")
                raise e
        
        # 동기 함수를 비동기로 실행
        documents_count, chunk_ids = await anyio.to_thread.run_sync(initialize_store)
        
        return InitializeVectorStoreResponse(
            status="successful",
            message=f"Vector store initialized successfully with {documents_count} documents",
            documents_count=documents_count,
            chunk_ids=chunk_ids
        )
        
    except Exception as e:
        logger.error(f"Vector store initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector store initialization failed: {str(e)}")

@router.get("/vector_store_status", response_model=VectorStoreStatusResponse)
async def get_vector_store_status():
    """
    벡터 스토어의 현재 상태를 확인합니다.
    """
    try:
        store = VectorStore(index_name=VECTOR_STORE_INDEX_NAME)
        index_exists = store.check_index_exists()
        
        return VectorStoreStatusResponse(
            status="successful",
            index_exists=index_exists,
            index_name=VECTOR_STORE_INDEX_NAME,
            message=f"Index '{VECTOR_STORE_INDEX_NAME}' {'exists' if index_exists else 'does not exist'}"
        )
    except Exception as e:
        logger.error(f"Failed to check vector store status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check vector store status: {str(e)}")

@router.get("/vector_store_stats")
async def get_vector_store_stats():
    """벡터 스토어 통계 정보 조회"""
    try:
        store = VectorStore(index_name=VECTOR_STORE_INDEX_NAME)
        stats = store.get_index_stats()
        return {
            "status": "successful",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Failed to get vector store stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/query_rag_by_category")
async def query_rag_by_category(input: QueryRagInput, category: str):
    """카테고리별 RAG 질의"""
    try:
        store = VectorStore(index_name=VECTOR_STORE_INDEX_NAME)
        results = store.similarity_search_by_category(input.prompt, category)
        
        # 결과를 orchestrator 형식으로 변환
        sources = []
        for doc in results:
            meta = dict(doc.metadata or {})
            sources.append(meta)
        
        return QueryRagResponse(
            status="successful", 
            response=f"Found {len(results)} results in category '{category}'",
            sources=sources
        )
    except Exception as e:
        logger.error(f"Category search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_rag_by_folder")
async def query_rag_by_folder(input: QueryRagInput, main_category: str, sub_category: str = None):
    """폴더별 RAG 질의 (강령, 법률, 상품 등)"""
    try:
        store = VectorStore(index_name=VECTOR_STORE_INDEX_NAME)
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
            status="successful",
            response=f"Found {len(results)} results in folder '{folder_path}'",
            sources=sources
        )
    except Exception as e:
        logger.error(f"Folder search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))