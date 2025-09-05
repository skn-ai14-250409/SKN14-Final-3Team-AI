from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any, Optional, Callable, Tuple
import math, json, re

# ------- LangChain -------
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
# from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser


# ------- Embeddings / Vector DB -------
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from dataclasses import dataclass
from langchain_core.documents import Document

from src.rag.vector_store import VectorStore
from src.config import PINECONE_KEY, MODEL_KEY, EMBEDDING_MODEL_NAME, EMBEDDING_BACKEND, VECTOR_STORE_INDEX_NAME, PINECONE_METRIC, MODEL_NAME


# 모델: 경량 + 결정적 라우팅 위해 temperature=0
llm = ChatOpenAI(
                model=MODEL_NAME,
                api_key=MODEL_KEY,
                temperature=0)

# ==============================================
# 1) 라우터 (고객응대 전용: LAW/RULE/PRODUCT)
# ==============================================
def parse_router_output(raw: str) -> str:
    """{"label": "..."} 형태만 안전 추출, 실패 시 PRODUCT"""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    txt = m.group(0) if m else raw
    try:
        data = json.loads(txt)
        label = (data.get("label") or "").upper().strip()
    except Exception:
        label = ""
    return label if label in {"LAW","RULE","PRODUCT"} else "PRODUCT"


router_prompt = ChatPromptTemplate.from_template(
    """You are a classifier for customer-facing banking queries.
Pick exactly one label from ["LAW","RULE","PRODUCT"] and return a compact JSON:
{{"label": "LAW" | "RULE" | "PRODUCT"}}

Guidelines:
- 법률/감독규정/시행령/시행규칙 중심 → "LAW"
- 은행 내부 업무처리 기준/내규 중심 → "RULE"
- 상품 안내/자격/한도/금리/수수료/절차 중심 → "PRODUCT"

User query: {query}
""")
# router_chain = LLMChain(llm=llm, prompt=router_prompt, output_parser=StrOutputParser())

router_chain = router_prompt | llm | StrOutputParser()





# ==============================================
# 2) LLM 체인 (knowledge / [LAW / RULE / PRODUCT])
# ==============================================
knowledge_prompt = ChatPromptTemplate.from_template(
"""당신은 한국 금융 전문가입니다.
금융 개념/용어/법규/감독규정 관련 질문에 간단하고 정확하게 답변하세요.
핵심 개념 정의와 예시를 3줄 내외로 제공합니다.
질문: {query}
""")

knowledge_chain = knowledge_prompt | llm # LLMChain(llm=llm, prompt=knowledge_prompt)



law_prompt = ChatPromptTemplate.from_template(
    """당신은 은행 창구 직원입니다. 고객에게 친절하게 설명하는 말투를 사용하세요.
질문이 법률, 감독규정, 시행령/시행규칙 등 법적 근거와 관련된 경우, 
관련 조문이나 규정을 가능한 한 명시하고, 이를 바탕으로 고객이 이해하기 쉽게 안내하세요.
- 먼저 규정 근거를 요약하고,
- 이어서 고객이 취해야 할 절차나 유의사항을 설명하세요.

질문: {query}


[RAG Answering Rules]
- If no relevant documents are retrieved from the RAG (evidence search), do not guess or make up an answer.  
- When evidence is insufficient, respond honestly and politely with a short message such as:  
  “I could not find related information. I will check and provide guidance later.”  
- Always make it clear when you do not know, so the customer does not receive incorrect or misleading information.  


# (선택) 법/감독규정 컨텍스트(근거 스니펫):
{{context}}



""")

law_chain = law_prompt|llm # LLMChain(llm=llm, prompt=law_prompt)

rule_prompt = ChatPromptTemplate.from_template(
    """당신은 은행 창구 직원입니다. 고객에게 친절하게 설명하는 말투를 사용하세요.
질문이 은행 내부 업무처리 기준, 지침, 내규와 관련된 경우, 
내규상의 제한, 승인 절차, 예외 사항 등을 고객이 알기 쉽게 풀어서 안내하세요.
- 단계별로 절차를 정리하고,
- 필요한 경우 준비해야 할 서류나 조건을 알려주세요.

질문: {query}


[RAG Answering Rules]
- If no relevant documents are retrieved from the RAG (evidence search), do not guess or make up an answer.  
- When evidence is insufficient, respond honestly and politely with a short message such as:  
  “I could not find related information. I will check and provide guidance later.”  
- Always make it clear when you do not know, so the customer does not receive incorrect or misleading information.  


# (선택) 내규 컨텍스트(근거 스니펫):
{{context}}"""
)

rule_chain = rule_prompt|llm # LLMChain(llm=llm, prompt=rule_prompt)

product_prompt = ChatPromptTemplate.from_template(
    """당신은 은행 창구 직원입니다. 고객에게 친절하게 설명하는 말투를 사용하세요.
질문이 상품 가입, 상환, 갈아타기, 금리, 수수료 등 상품 정보와 관련된 경우, 
상품의 특징과 조건을 고객 친화적인 언어로 설명하세요.
- 자격 요건, 한도, 금리, 수수료 등을 간단히 짚고,
- 절차를 체크리스트로 안내하며,
- 다음에 취할 행동을 제안하세요.

질문: {query}


[RAG Answering Rules]
- If no relevant documents are retrieved from the RAG (evidence search), do not guess or make up an answer.  
- When evidence is insufficient, respond honestly and politely with a short message such as:  
  “I could not find related information. I will check and provide guidance later.”  
- Always make it clear when you do not know, so the customer does not receive incorrect or misleading information.  


# (선택) 상품 컨텍스트(근거 스니펫):
{{context}}"""
)

product_chain = product_prompt|llm # LLMChain(llm=llm, prompt=product_prompt)

# ==============================================
# 3) Pinecone 초기화 + 임베딩
# ==============================================
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(VECTOR_STORE_INDEX_NAME)

vs = VectorStore(EMBEDDING_BACKEND)
embedding_model = vs._get_embedding_model()




# ==============================================
# 4) 점수 정규화 & 신뢰도 라벨링
# ==============================================
def normalize_score(raw: float, metric: Literal["cosine","dotproduct","euclidean"]) -> float:
    """모든 metric을 0~1(높을수록 유사)로 정규화"""
    if metric == "cosine":
        return max(0.0, min(1.0, float(raw)))
    elif metric == "dotproduct":
        return 1.0 / (1.0 + math.exp(-float(raw)))  # 로지스틱 스쿼시(경험적)
    elif metric == "euclidean":
        sigma = 5.0  # 데이터 기반으로 조정 권장
        return math.exp(-(float(raw)**2) / (2 * sigma**2))
    return float(raw)

def label_confidence(norm: float) -> Literal["high","medium","low"]:
    if norm >= 0.85: return "high"
    if norm >= 0.70: return "medium"
    return "low"

# ==============================================
# 5) Evidence 구조 & 검색 함수
# ==============================================
class Evidence(TypedDict, total=False):
    doc_id: str
    source: str
    page: Optional[int]
    chunk_id: Optional[int]
    text: str
    raw_score: float
    norm_score: float
    level: Literal["high","medium","low"]

def fetch_evidences(category: str, query: str, k: int = 5) -> List[Evidence]:
    """Pinecone namespace=category(LAW/RULE/PRODUCT)에서 top-k 검색 → Evidence 배열 반환"""
    if EMBEDDING_BACKEND == "openai":
        vec = embedding_model.embed_query(query)
    else:
        vec = embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
    res = index.query(vector=vec, top_k=k, include_metadata=True, namespace=category)

    evidences: List[Evidence] = []
    for match in res.get("matches", []):
        md: Dict[str, Any] = match.get("metadata", {}) or {}
        raw = float(match["score"])
        norm = normalize_score(raw, PINECONE_METRIC)
        evidences.append(Evidence(
            doc_id = str(md.get("doc_id") or md.get("file_id") or md.get("id") or ""),
            source = str(md.get("source") or md.get("file_name") or category),
            page = md.get("page_num") or md.get("page") or None,
            chunk_id = md.get("chunk_index") or md.get("chunk_id") or None,
            text = md.get("text",""),
            raw_score = raw,
            norm_score = norm,
            level = label_confidence(norm),
        ))
    return evidences

def build_context_for_llm(evidences: List[Evidence], max_chars: int = 1200) -> str:
    """LLM 주입용 컨텍스트: 텍스트만 깔끔히 묶어서 길이 제한"""
    lines, total = [], 0
    for ev in evidences:
        snippet = (ev.get("text") or "").strip()
        if not snippet:
            continue
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        line = f"- {snippet}"
        total += len(line)
        if total > max_chars:
            break
        lines.append(line)
    return "\n".join(lines)

# ==============================================
# 6) Orchestrator
# - mode: "knowledge" | "customer"
# - 반환: {"answer": str, "evidences": List[Evidence], "category": Optional[str]}
# ==============================================
NO_ANSWER_MSG = "해당 정보를 찾을 수 없습니다."

def query_rag(self, query: str) -> Dict[str, Any]:
        """RAG를 사용하여 쿼리에 응답합니다."""
        self.vector_store.get_index_ready()
        raw_retrieved = self.vector_store.similarity_search(query)

        # 검색 결과 정규화
        retrieved_docs = self._normalize_retrieved(raw_retrieved)

        # 컨텍스트가 비었으면 바로 종료
        if not retrieved_docs:
            return {"response": NO_ANSWER_MSG, "sources": []}

        context_text = self._format_context(retrieved_docs)
        if not context_text.strip():
            return {"response": NO_ANSWER_MSG, "sources": []}
        
        # 검색된 문서의 최소 길이 체크 (너무 짧으면 관련성이 낮을 가능성)
        if len(context_text) < 50:
            return {"response": NO_ANSWER_MSG, "sources": []}

        # LLM 호출
        prompt = [self._build_system_message(retrieved_docs), HumanMessage(query)]
        try:
            response: str = self.slm.invoke(prompt)
        except Exception:
            response = None

        # 소스 메타데이터 추출
        sources = []
        for d in retrieved_docs:
            meta = dict(d.metadata or {})
            if "relative_path" not in meta:
                rf = meta.get("root_folder")
                fn = meta.get("file_name")
                if rf and fn:
                    meta["relative_path"] = f"{rf}/{fn}"
            sources.append(meta)

        return {
            "response": response or NO_ANSWER_MSG,
            "sources": sources
        }


def _fallback_to_query_rag(query: str, label: Optional[str]) -> Dict[str, Any]:
    """query_rag() 결과를 run_workflow 반환 스키마로 변환"""
    try:
        r = query_rag(query)  # ← 네가 준 query_rag를 그대로 호출
    except Exception:
        r = {"response": NO_ANSWER_MSG, "sources": []}
    return {
        "answer": r.get("response", NO_ANSWER_MSG),
        "evidences": r.get("sources", []),
        "category": label
    }


def run_workflow(mode: str, query: str) -> Dict[str, Any]:
    mode = (mode or "").strip().lower()

    if mode == "knowledge":
        answer = knowledge_chain.invoke({"query":query})
        return {"answer": answer, "evidences": [], "category": None}

    if mode == "customer":
        # 1) 고객응대 하위 라우팅
        raw = router_chain.invoke({"query": query})
        label = parse_router_output(raw)  # "LAW" | "RULE" | "PRODUCT"


        # 2) 검색 → 컨텍스트 생성
        evs = fetch_evidences(label, query, k=5)

        # (선택) 너무 낮은 norm_score는 LLM 컨텍스트에서 제외하고 UI에는 노출 가능
        used_for_ctx = [e for e in evs if e["norm_score"] >= 0.00]
        ctx = build_context_for_llm(used_for_ctx)

        # 3) 생성
        if label == "LAW":
            answer = law_chain.invoke({"query": query, "context": ctx})
        elif label == "RULE":
            answer = rule_chain.invoke({"query": query, "context": ctx})
        else:
            answer = product_chain.invoke({"query": query, "context": ctx})

        # 4) 반환(답변 + 전체 evidences: FE에서 뱃지/툴팁/정렬에 활용)
        return {"answer": answer, "evidences": evs, "category": label}

    # 모드 오류
    return {"answer": "모드가 지정되지 않았습니다. 'knowledge' 또는 'customer'를 선택해 주세요.", "evidences": [], "category": None}


# def run_workflow(mode: str, query: str) -> Dict[str, Any]:
#     mode = (mode or "").strip().lower()

#     if mode == "knowledge":
#         answer = knowledge_chain.invoke({"query":query})
#         # ★ 지식 모드에서도 LLM 실패/빈 응답이면 RAG 폴백
#         if not (answer and str(getattr(answer, "content", answer)).strip()):
#             return _fallback_to_query_rag(query, None)
#         return {"answer": answer, "evidences": [], "category": None}

#     if mode == "customer":
#         # 1) 고객응대 하위 라우팅
#         raw = router_chain.invoke({"query": query})
#         label = parse_router_output(raw)  # "LAW" | "RULE" | "PRODUCT"

#         # 2) 검색 → 컨텍스트 생성
#         evs = fetch_evidences(label, query, k=5)

#         used_for_ctx = [e for e in evs if e["norm_score"] >= 0.00]
#         ctx = build_context_for_llm(used_for_ctx)

#         # ★ 컨텍스트가 비었거나 너무 짧으면 즉시 RAG 폴백
#         if not ctx or not ctx.strip() or len(ctx) < 50:
#             return _fallback_to_query_rag(query, label)

#         # 3) 생성
#         if label == "LAW":
#             answer = law_chain.invoke({"query": query, "context": ctx})
#         elif label == "RULE":
#             answer = rule_chain.invoke({"query": query, "context": ctx})
#         else:
#             answer = product_chain.invoke({"query": query, "context": ctx})

#         # ★ 생성 결과가 비어 있으면 RAG 폴백
#         if not (answer and str(getattr(answer, "content", answer)).strip()):
#             return _fallback_to_query_rag(query, label)

#         # 4) 반환(답변 + 전체 evidences: FE에서 뱃지/툴팁/정렬에 활용)
#         return {"answer": answer, "evidences": evs, "category": label}

#     # 모드 오류
#     return {
#         "answer": "모드가 지정되지 않았습니다. 'knowledge' 또는 'customer'를 선택해 주세요.",
#         "evidences": [],
#         "category": None
#     }


# ==============================================
# 7) Quick test
# ==============================================
if __name__ == "__main__":
    q_k = "원리금균등상환과 원금균등상환의 차이를 알려줘."
    q_c1 = "전세자금대출 갈아타기 가능한가요? 절차와 필요서류 알려주세요."
    q_c2 = "신용대출 한도와 금리 산정 기준이 궁금합니다. 내규상 제한이 있나요?"
    q_c3 = "적금 중도해지 수수료는 어떤 법 규정에 따르나요?"
    q_c4 = "햇살론 가입하고 싶습니다. 가입 조건 알려주세요"

    print("\n[knowledge]")
    print(run_workflow("knowledge", q_k)["answer"].content)

    print("\n[customer → PRODUCT 예시 기대]")
    out1 = run_workflow("customer", q_c1)
    print(out1["category"], out1["answer"].content)
    # print(out1["evidences"])  # FE에서 사용

    print("\n[customer → RULE/Law 후보]")
    out2 = run_workflow("customer", q_c2)
    print(out2["category"], out2["answer"].content)

    print("\n[customer → LAW 후보]")
    out3 = run_workflow("customer", q_c3)
    print(out3["category"], out3["answer"].content)

    print("\n[customer → customer 예시 기대]")
    out4 = run_workflow("customer", q_c4)
    print(out4["category"], out4["answer"])
