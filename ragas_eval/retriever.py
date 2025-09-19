from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    """질문→컨텍스트 리스트를 반환하는 인터페이스(필수)"""

    @abstractmethod
    def retrieve(self, question: str, k: int = 4) -> List[str]:
        ...


class EchoRetriever(BaseRetriever):
    """
    테스트용: '정답 텍스트'를 컨텍스트처럼 사용.
    실제에선 Pinecone/FAISS/ES로 교체.
    """

    def __init__(self, refs: List[str]):
        self.refs = refs  # questions와 같은 순서의 정답 리스트

    def retrieve(self, question: str, k: int = 4) -> List[str]:
        # 단순히 해당 질문의 정답을 k개 복제 → 자리표시자
        # (실전: 여기에 Vector 검색을 구현)
        # 주의: 이 구현은 외부 인덱스를 안 쓰므로 '질문별' 매핑이 필요 → 외부에서 zip으로 묶어 호출 권장
        raise NotImplementedError("EchoRetriever는 batch 사용 전제로, 아래 헬퍼를 사용해줘.")


def build_echo_contexts(refs: List[str], k: int = 4) -> List[List[str]]:
    """정답을 context로 복제하는 간단 헬퍼 (테스트/샘플 파이프라인 용)"""
    ctxs: List[List[str]] = []
    for r in refs:
        snippet = r if len(r) <= 1200 else r[:1200]
        ctxs.append([snippet] * k)
    return ctxs

