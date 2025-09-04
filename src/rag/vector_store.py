from typing import List, Literal, Optional, Dict, Any
from uuid import uuid4
import argparse
import logging

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from .document_loader import DocumentLoader

from src.config import PINECONE_KEY, MODEL_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_BACKEND, VECTOR_STORE_INDEX_NAME

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, index_name: str, embedding_backend: Literal["openai", "huggingface"] = "openai"):
        self.pc = Pinecone(api_key=PINECONE_KEY)
        self.index_name: str = VECTOR_STORE_INDEX_NAME
        self.embedding_backend: str = EMBEDDING_BACKEND

    def create_index(self) -> None:
        dimension = 1536 if self.embedding_backend == "openai" else 1024
        # This uses the pinecone library
        index = self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f'Created index: {index}')

    def check_index_exists(self) -> bool:
        # This uses the pinecone library
        if not self.pc.has_index(self.index_name):
            return False
        return True
    
    def get_index_ready(self) -> None:
        if not self.check_index_exists():
            self.create_index()
    
    def _get_embedding_model(self):
        """embedding_backend 설정에 맞는 embedding model 반환"""
        if self.embedding_backend == "openai":
            return OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME,
                api_key=MODEL_KEY
            )
        elif self.embedding_backend == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        else:
            raise ValueError(f"Unknown embedding backend: {self.embedding_backend}")



    def get_index(self) -> PineconeVectorStore:
        # This uses the Langchain-pinecone library
        embedding_model = self._get_embedding_model()
        index = self.pc.Index(self.index_name)
        vs_index = PineconeVectorStore(index=index, embedding=embedding_model)
        return vs_index

    def add_documents_to_index(self, documents: List[Document]) -> List[str]:
        """문서를 인덱스에 추가 (개선된 메타데이터 포함) - 배치 처리"""
        vs_index: PineconeVectorStore = self.get_index()
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # 메타데이터 정리 - Pinecone에서 지원하지 않는 타입 제거
        cleaned_documents = []
        for doc in documents:
            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                # Pinecone은 string, number, boolean, list of strings만 지원
                if isinstance(value, (str, int, float, bool)):
                    cleaned_metadata[key] = value
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    cleaned_metadata[key] = value
                else:
                    # 복잡한 타입은 문자열로 변환
                    cleaned_metadata[key] = str(value)
            
            cleaned_doc = Document(
                page_content=doc.page_content,
                metadata=cleaned_metadata
            )
            cleaned_documents.append(cleaned_doc)
        
        # 배치 처리 - OpenAI API 토큰 제한 대응 (배치당 100개 문서)
        batch_size = 100
        all_ids = []
        
        for i in range(0, len(cleaned_documents), batch_size):
            batch_docs = cleaned_documents[i:i + batch_size]
            batch_ids = uuids[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(cleaned_documents) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)")
            
            try:
                batch_result_ids: List[str] = vs_index.add_documents(documents=batch_docs, ids=batch_ids)
                all_ids.extend(batch_result_ids)
                logger.info(f"Successfully added batch of {len(batch_result_ids)} documents")
            except Exception as e:
                logger.error(f"Failed to add batch starting at index {i}: {e}")
                # 배치가 실패해도 계속 진행
                continue
        
        logger.info(f"Added {len(all_ids)} documents to index (total attempted: {len(documents)})")
        return all_ids
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """향상된 유사도 검색 - 메타데이터 필터링 지원"""
        vs_index: PineconeVectorStore = self.get_index()
        
        if filter_dict:
            # Pinecone 필터 형식으로 변환
            results = vs_index.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = vs_index.similarity_search(query, k=k)
        
        return results

    def similarity_search_by_category(self, query: str, category: str, k: int = 5) -> List[Document]:
        """카테고리별 검색"""
        filter_dict = {"document_category": category}
        return self.similarity_search(query, k=k, filter_dict=filter_dict)

    def similarity_search_by_product_type(self, query: str, product_type: str, k: int = 5) -> List[Document]:
        """상품 유형별 검색"""
        filter_dict = {"product_type": product_type}
        return self.similarity_search(query, k=k, filter_dict=filter_dict)

    def similarity_search_by_main_category(self, query: str, main_category: str, k: int = 5) -> List[Document]:
        """메인 카테고리별 검색 (강령, 법률, 상품 등)"""
        filter_dict = {"main_category": main_category}
        return self.similarity_search(query, k=k, filter_dict=filter_dict)

    def similarity_search_by_folder(self, query: str, main_category: str, sub_category: str = None, k: int = 5) -> List[Document]:
        """폴더 구조 기반 검색"""
        filter_dict = {"main_category": main_category}
        if sub_category:
            filter_dict["sub_category"] = sub_category
        return self.similarity_search(query, k=k, filter_dict=filter_dict)
    

    def similarity_search_with_metadata(self, query: str, k: int = 5,
                                       filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """향상된 메타데이터와 함께 유사도 검색 (개선된 구조 지원)"""
        vs_index: PineconeVectorStore = self.get_index()
        
        if filter_dict:
            docs_with_scores = vs_index.similarity_search_with_score(query=query, k=k, filter=filter_dict)
        else:
            docs_with_scores = vs_index.similarity_search_with_score(query=query, k=k)

        sources = []
        for doc, score in docs_with_scores:
            meta = doc.metadata or {}
            sources.append({
                # === 핵심 식별 정보 ===
                "file_name": meta.get("file_name"),
                "file_path": meta.get("file_path"),
                "relative_path": meta.get("file_path"),  # 하위 호환성
                
                # === 폴더 구조 정보 (개선된 부분) ===
                "main_category": meta.get("main_category"),      # 강령, 법률, 상품
                "sub_category": meta.get("sub_category"),        # 공통, 개인_신용대출
                
                # === 문서 분류 ===
                "document_category": meta.get("document_category"), # policy, product
                "subcategory": meta.get("subcategory"),            # ethics, personal_loan
                "business_unit": meta.get("business_unit"),        # retail_banking
                
                # === 청크 정보 ===
                "chunk_index": meta.get("chunk_index"),
                "page_number": meta.get("page_number"),
                "content_length": meta.get("content_length"),
                
                # === 상품 정보 (해당하는 경우) ===
                "product_type": meta.get("product_type"),
                "target_customer": meta.get("target_customer"),
                
                # === 검색 최적화 ===
                "keywords": meta.get("keywords", []),
                "tags": meta.get("tags", []),
                "score": float(score) if score is not None else None,
                
                # === 컨텐츠 기반 태그 ===
                "contains_interest_rate": meta.get("contains_interest_rate", False),
                "contains_conditions": meta.get("contains_conditions", False),
                "contains_application_info": meta.get("contains_application_info", False),
                "contains_policy": meta.get("contains_policy", False),
                "contains_ethics": meta.get("contains_ethics", False),
                
                # === 원본 메타데이터 보존 ===
                "metadata": meta,
            })
        return sources

    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스 통계 정보 조회"""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def delete_documents_by_filter(self, filter_dict: Dict[str, Any]) -> None:
        """필터 조건에 맞는 문서들 삭제"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(filter=filter_dict)
            logger.info(f"Deleted documents with filter: {filter_dict}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def delete_vectors_by_condition(self, field: str, value: str) -> int:
        """특정 조건에 맞는 벡터만 삭제"""
        try:
            index = self.pc.Index(self.index_name)
            
            # 먼저 해당 조건에 맞는 벡터들을 검색하여 개수 확인
            query_vector = [0.0] * (1536 if self.embedding_backend == "openai" else 1024)  # 더미 벡터
            results = index.query(
                vector=query_vector,
                filter={field: {"$eq": value}},
                top_k=10000,  # 충분히 큰 값
                include_metadata=True
            )
            
            if not results.matches:
                logger.info(f"No vectors found with {field} = {value}")
                return 0
            
            # 삭제할 벡터 ID들 수집
            ids_to_delete = [match.id for match in results.matches]
            deleted_count = len(ids_to_delete)
            
            # 벡터 삭제
            index.delete(ids=ids_to_delete)
            logger.info(f"Deleted {deleted_count} vectors with {field} = {value}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete vectors by condition: {e}")
            raise
    

    def delete_all_vectors(self) -> None:
        vs_index: PineconeVectorStore = self.get_index()
        return vs_index.delete(delete_all=True)


if __name__ == "__main__":
    data_folder = r"C:\Workspaces\SKN14-Final-3Team\SKN14-Final-3Team-private\SKN14-Final-3Team-Data2"
    
    # 1. 문서 로드
    docs = DocumentLoader.process_folder_and_get_chunks(data_folder)
    print(f"Loaded {len(docs)} documents from {data_folder}")
    
    # 2. VectorStore 초기화
    store = VectorStore(index_name= VECTOR_STORE_INDEX_NAME, embedding_backend=EMBEDDING_BACKEND)
    store.get_index_ready()
    
    # 3. Pinecone에 문서 추가
    ids = store.add_documents_to_index(docs)
    print(f"Added {len(ids)} documents to Pinecone index")