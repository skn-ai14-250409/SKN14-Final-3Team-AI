from typing import Literal, List

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from src.orchestrator import Orchestrator

router = APIRouter()

class BaseResponse(BaseModel):
    status: Literal["successful", "fail"]

class QueryRagInput(BaseModel):
    prompt: str

class QueryRagResponse(BaseResponse):
    response: str

class RunWorkflowInput(BaseModel):
    prompt: str

class RunWorkflowResponse(BaseResponse):
    response: str

class UploadRagDocsResponse(BaseResponse):
    chunk_ids: List[str]

@router.get("/healthcheck")
async def healthcheck():
    return BaseResponse(status="successful")

@router.post("/query_rag")
async def query_rag(input: QueryRagInput):
    orchestrator = Orchestrator()
    response = orchestrator.query_rag(input.prompt)
    return QueryRagResponse(
        status="successful",
        response=response
    )

@router.delete("/delete_all_vectors")
async def delete_all_vectors():
    orchestrator = Orchestrator()
    orchestrator.delete_all_vectors()
    return BaseResponse(status="successful")

@router.post("/run_worflow")
async def run_workflow(input: RunWorkflowInput):
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