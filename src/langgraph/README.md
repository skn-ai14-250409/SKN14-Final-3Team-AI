# LangGraph RAG Workflow Documentation

## 📁 폴더 구조 개요

```
src/langgraph/
├── langgraph_v2.py          # 메인 진입점 (호환성 유지)
├── workflow.py              # 워크플로우 클래스 정의
├── workflow_factory.py      # 워크플로우 팩토리 함수들
├── state.py                 # 상태 정의 (RAGState)
├── nodes.py                 # 노드 함수들 (실제 구현)
├── tools.py                 # 도구 정의 (LangChain @tool)
├── session_manager.py       # 세션 관리 클래스
├── utils.py                 # 공통 유틸리티 함수들
├── guardrails/              # 가드레일 설정 파일들
│   ├── glossary_terms.yaml
│   └── policy_rules.yaml
└── README.md               # 이 문서
```

## 🎯 전체 아키텍처

### **LangGraph V2 모듈화 구조**
- **툴콜링 기반 워크플로우**: LLM이 도구를 선택하여 실행하는 구조
- **모듈화된 설계**: 각 기능별로 파일을 분리하여 유지보수성 향상
- **세션 관리**: 대화 히스토리와 컨텍스트 추적
- **멀티턴 대화**: 연속적인 대화 지원
- **SLM 인스턴스 최적화**: 중복 생성 제거로 성능 향상

## 📋 각 파일 상세 설명

### 1. `langgraph_v2.py` - 메인 진입점
**역할**: 모듈화된 구조의 통합 진입점
**주요 기능**:
- 모든 모듈을 import하여 통합
- `get_langgraph_workflow()` 함수 제공
- 싱글톤 패턴으로 워크플로우 인스턴스 관리
- 기존 코드와의 호환성 유지

```python
# 사용 예시
from src.langgraph.langgraph_v2 import get_langgraph_workflow
workflow = get_langgraph_workflow()
result = workflow.run_workflow("햇살론에 대해 알려주세요", "session_123")
```

### 2. `workflow.py` - 워크플로우 클래스
**역할**: RAG 워크플로우의 핵심 실행 클래스
**주요 기능**:
- `LangGraphRAGWorkflow` 클래스 정의
- `run_workflow()` 메서드로 전체 워크플로우 실행
- 세션 관리와 대화 히스토리 처리
- 에러 처리 및 로깅

```python
class LangGraphRAGWorkflow:
    def __init__(self, checkpointer=None, llm=None):
        self.slm = SLM()
        self.vector_store = VectorStore()
        self.workflow = create_rag_workflow(checkpointer=checkpointer, llm=llm)
    
    def run_workflow(self, query: str, session_id: str = None) -> Dict[str, Any]:
        # 워크플로우 실행 로직
```

### 3. `workflow_factory.py` - 워크플로우 팩토리
**역할**: 워크플로우 생성 및 구성
**주요 기능**:
- `create_rag_workflow()` 함수로 워크플로우 생성
- SLM 인스턴스 생성 및 모든 노드에 전달 (중복 제거)
- 노드 간 연결 및 라우팅 설정
- 워크플로우 컴파일 및 반환

```python
def create_rag_workflow(checkpointer=None, llm=None) -> StateGraph:
    # SLM 인스턴스 생성 (중복 제거)
    slm = SLM()
    
    # 각 노드에 SLM 전달
    intent_classification_with_slm = partial(intent_classification_node, slm=slm)
    chitchat_with_slm = partial(chitchat_node, slm=slm)
    # ... 모든 노드에 적용
```

### 4. `state.py` - 상태 정의
**역할**: 워크플로우 상태 스키마 정의
**주요 기능**:
- `RAGState` TypedDict로 상태 구조 정의
- 모든 노드 간 공유되는 데이터 구조
- 타입 안전성 보장

```python
class RAGState(TypedDict):
    query: str
    response: str
    session_context: Optional[SessionContext]
    conversation_history: List[ConversationTurn]
    # ... 기타 상태 필드들
```

### 5. `nodes.py` - 노드 함수들
**역할**: 워크플로우의 각 단계별 실행 함수들
**주요 기능**:
- 11개의 노드 함수 정의
- SLM 인스턴스를 매개변수로 받아 중복 생성 방지
- 각 노드별 특화된 로직 구현
- 에러 처리 및 로깅

#### **노드 함수 목록**:
1. `session_init_node()` - 세션 초기화
2. `intent_classification_node()` - 의도 분류 (4개 카테고리)
3. `supervisor_node()` - 중앙 관리자 (툴 선택)
4. `chitchat_node()` - 일반 대화 처리
5. `general_faq_node()` - 일반 은행 FAQ
6. `product_extraction_node()` - 상품명 추출
7. `product_search_node()` - 상품 검색
8. `session_summary_node()` - 세션 요약 생성
9. `rag_search_node()` - RAG 검색
10. `guardrail_check_node()` - 가드레일 검사
11. `answer_node()` - 최종 답변 생성

```python
def chitchat_node(state: RAGState, slm: SLM = None) -> RAGState:
    """일반 대화 처리"""
    # SLM 인스턴스 생성 (매개변수로 받지 않은 경우에만)
    if slm is None:
        slm = SLM()
    # ... 노드 로직
```

### 6. `tools.py` - 도구 정의
**역할**: LangChain @tool 데코레이터로 정의된 도구들
**주요 기능**:
- 9개의 도구 함수 정의
- 각 도구의 스키마와 설명 제공
- supervisor_node에서 사용 가능한 도구 목록

#### **도구 목록**:
1. `chitchat()` - 일반 대화
2. `general_faq()` - 일반 FAQ
3. `rag_search()` - RAG 검색
4. `product_extraction()` - 상품명 추출
5. `product_search()` - 상품 검색
6. `session_summary()` - 세션 요약
7. `guardrail_check()` - 가드레일 검사
8. `answer()` - 답변 생성
9. `intent_classification()` - 의도 분류

```python
@tool(parse_docstring=True)
def chitchat(thought: str, query: str):
    """
    Handle casual conversation and greetings.
    
    Args:
        thought: Reasoning for choosing this tool
        query: User's casual conversation or greeting
    """
```

### 7. `session_manager.py` - 세션 관리
**역할**: 대화 세션 및 히스토리 관리
**주요 기능**:
- `SessionManager` 클래스로 세션 생성/관리
- 대화 히스토리 저장 및 조회
- 세션 만료 처리
- 컨텍스트 추적

```python
class SessionManager:
    def create_session(self, session_id: str = None) -> SessionContext:
        # 세션 생성 로직
    
    def get_session(self, session_id: str) -> Optional[SessionContext]:
        # 세션 조회 로직
```

### 8. `utils.py` - 공통 유틸리티
**역할**: 공통 함수 및 상수 정의
**주요 기능**:
- 시스템 프롬프트 템플릿
- 에러 메시지 정의
- 공통 유틸리티 함수들
- 상수 정의

```python
# 시스템 프롬프트
SYSTEM_PROMPTS = {
    "rag_system": "당신은 KB금융그룹의 전문 상담사입니다...",
    "faq_system": "일반적인 은행 FAQ 질문에 대해...",
    # ... 기타 프롬프트들
}

# 에러 메시지
ERROR_MESSAGES = {
    "general_error": "죄송합니다. 처리 중 오류가 발생했습니다.",
    # ... 기타 에러 메시지들
}
```

## 🔄 워크플로우 실행 흐름

### **1단계: 세션 초기화**
```
session_init_node → 세션 생성/조회 → 대화 히스토리 로드
```

### **2단계: 의도 분류**
```
intent_classification_node → 4개 카테고리 분류:
├── general_banking_FAQs (일반 은행 FAQ)
├── industry_policies_and_regulations (업계 규제)
├── company_rules (회사 내규)
└── company_products (회사 상품)
```

### **3단계: 중앙 관리자**
```
supervisor_node → 툴 선택 → 다음 노드로 라우팅
```

### **4단계: 특화 노드 실행**
```
├── chitchat_node (일반 대화)
├── general_faq_node (일반 FAQ)
├── product_extraction_node → product_search_node (상품 검색)
├── session_summary_node (세션 요약)
├── rag_search_node (RAG 검색)
└── guardrail_check_node (가드레일 검사)
```

### **5단계: 최종 답변**
```
answer_node → 응답 생성 → 세션 업데이트 → 종료
```

## 🚀 성능 최적화

### **SLM 인스턴스 관리**
- **이전**: 각 노드마다 SLM 인스턴스 생성 (11번)
- **이후**: workflow_factory에서 1번만 생성하고 모든 노드에 전달
- **효과**: 메모리 사용량 감소, 초기화 시간 단축

### **모듈화 설계**
- 각 기능별 파일 분리로 유지보수성 향상
- 공통 유틸리티 함수 중앙화
- 타입 안전성 보장

## 📊 의도 분류 시스템

### **4개 카테고리**
1. **general_banking_FAQs**: 일반적인 은행 개념이나 금융 상식
   - 예: "예금이 뭐예요?", "적금이 뭐예요?"

2. **industry_policies_and_regulations**: 은행업 규제 및 정책
   - 예: "KYC 규정이 뭐예요?", "바젤3 규정에 대해"

3. **company_rules**: KB금융그룹 내부 규칙 및 정책
   - 예: "직원 휴가 정책은 어떻게 되나요?"

4. **company_products**: KB금융그룹의 구체적인 상품
   - 예: "햇살론 대출에 대해", "KB카드 혜택이 뭐예요?"

## 🔧 사용법

### **기본 사용법**
```python
from src.langgraph.langgraph_v2 import get_langgraph_workflow

# 워크플로우 인스턴스 가져오기
workflow = get_langgraph_workflow()

# 쿼리 실행
result = workflow.run_workflow(
    query="햇살론 대출에 대해 알려주세요",
    session_id="user_123"
)

print(result["response"])
```

### **FastAPI에서 사용**
```python
from src.langgraph.langgraph_v2 import get_langgraph_workflow

@app.post("/api/v1/langgraph/langgraph_rag")
async def langgraph_rag(request: LangGraphRequest):
    workflow = get_langgraph_workflow()
    result = workflow.run_workflow(
        query=request.prompt,
        session_id=request.session_id
    )
    return result
```

## 🐛 에러 처리

### **공통 에러 처리**
- 모든 노드에서 `try-except` 블록으로 에러 처리
- `create_error_response()` 함수로 일관된 에러 응답
- 로깅을 통한 디버깅 지원

### **에러 메시지**
```python
ERROR_MESSAGES = {
    "general_error": "죄송합니다. 처리 중 오류가 발생했습니다.",
    "search_error": "죄송합니다. 검색 중 오류가 발생했습니다.",
    "no_documents": "죄송합니다. 관련 문서를 찾을 수 없습니다.",
    # ... 기타 에러 메시지들
}
```

## 📝 로깅 시스템

### **노드별 로깅**
- 각 노드 실행 시 이모지와 함께 로깅
- PDF 문서 정보 로깅 (파일명, 페이지 번호)
- 의도 분류 결과 로깅

### **로그 예시**
```
🎯 [NODE] intent_classification_node 실행 시작
📚 [NODE] rag_search_node 실행 시작
📄 [RAG] 사용된 PDF 문서 정보:
  📋 문서 1: 햇살론_대출_상품안내.pdf (페이지: 3)
  📋 문서 2: KB카드_혜택안내.pdf (페이지: 1)
```

## 🔄 멀티턴 대화

### **세션 관리**
- `SessionManager`로 대화 히스토리 관리
- 컨텍스트 유지를 통한 연속 대화 지원
- 세션 만료 및 자동 정리

### **대화 흐름**
```
첫 번째 질문 → 세션 생성 → 의도 분류 → 답변 생성
두 번째 질문 → 기존 세션 로드 → 컨텍스트 고려 → 답변 생성
```

## 🛡️ 가드레일 시스템

### **응답 검증**
- `guardrail_check_node`에서 응답 검증
- 금융 규제 준수 확인
- 부적절한 내용 필터링

### **가드레일 파일**
- `guardrails/glossary_terms.yaml`: 금융 용어 정의
- `guardrails/policy_rules.yaml`: 정책 규칙 정의

## 📈 성능 모니터링

### **메트릭**
- 노드 실행 시간 측정
- 메모리 사용량 모니터링
- 에러율 추적

### **최적화 포인트**
- SLM 인스턴스 재사용
- 벡터 검색 최적화
- 캐싱 전략 적용

---

## 🎯 요약

이 `langgraph` 폴더는 **모듈화된 RAG 워크플로우**를 제공하며, 다음과 같은 특징을 가집니다:

- ✅ **툴콜링 기반**: LLM이 도구를 선택하여 실행
- ✅ **모듈화 설계**: 각 기능별 파일 분리
- ✅ **성능 최적화**: SLM 인스턴스 중복 생성 제거
- ✅ **멀티턴 대화**: 세션 관리로 연속 대화 지원
- ✅ **의도 분류**: 4개 카테고리로 정확한 라우팅
- ✅ **에러 처리**: 견고한 에러 처리 및 로깅
- ✅ **가드레일**: 금융 규제 준수 및 응답 검증

이 구조를 통해 **확장 가능하고 유지보수하기 쉬운** RAG 시스템을 구축할 수 있습니다.