# 🧪 LangGraph 실험용 RAG 시스템

이 폴더는 기존의 `orchestrator.py`와 동일한 기능을 **LangGraph**로 구현한 실험용 코드입니다.

## 📋 목적

- 기존 코드는 **그대로 유지**하면서 새로운 접근 방식 테스트
- LangGraph의 그래프 기반 워크플로우 경험
- 성능 및 가독성 비교
- 미래 확장 가능성 탐색

## 🏗️ 구조

### `langgraph_rag.py`
- **LangGraphRAGWorkflow**: 메인 워크플로우 클래스
- **RAGState**: 그래프 상태 관리
- **노드들**: 각 처리 단계를 개별 함수로 분리

### 워크플로우 그래프
```
START → classify_intent → [조건부 분기]
                         ├─ general_faq → handle_general_faq → END
                         └─ rag_needed  → extract_product_name 
                                        → search_documents 
                                        → filter_relevance 
                                        → generate_response → END
```

## 🚀 사용법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. API 엔드포인트 사용
```bash
# 기존 방식 (그대로 유지)
POST /query_rag
{
  "prompt": "KB 스마트론에 대해 알려주세요"
}

# 실험용 LangGraph 방식
POST /experimental/langgraph_rag
{
  "prompt": "KB 스마트론에 대해 알려주세요"
}
```

### 3. 응답 비교
**기존 방식:**
```json
{
  "status": "successful",
  "response": "KB 스마트론은...",
  "sources": [...]
}
```

**LangGraph 방식:**
```json
{
  "status": "successful",
  "response": "KB 스마트론은...",
  "sources": [...],
  "category": "company_products",
  "experimental": true,
  "workflow_type": "langgraph"
}
```

## 🔍 주요 차이점

| 측면 | 기존 Orchestrator | LangGraph 실험 |
|------|------------------|----------------|
| **구조** | 클래스 메서드 기반 | 그래프 노드 기반 |
| **상태 관리** | 지역 변수 | 중앙화된 State |
| **흐름 제어** | if/else 조건문 | 조건부 엣지 |
| **디버깅** | print 로그 | 그래프 시각화 가능 |
| **확장성** | 메서드 추가 | 노드/엣지 추가 |
| **복잡성** | 단순함 | 학습 곡선 |

## 🎯 LangGraph의 장점

### ✅ **모듈성**
- 각 처리 단계가 독립적인 노드
- 재사용 가능한 컴포넌트

### ✅ **가시성**
- 워크플로우를 그래프로 시각화 가능
- 실행 경로 추적 용이

### ✅ **확장성**
- 새로운 노드 추가 용이
- 복잡한 분기 로직 구현 가능

### ✅ **상태 관리**
- 중앙화된 상태로 데이터 흐름 명확
- 디버깅 시 상태 검사 용이

## ⚠️ LangGraph의 단점

### ❌ **복잡성 증가**
- 단순한 작업에 오버엔지니어링
- 학습 곡선 존재

### ❌ **성능 오버헤드**
- 그래프 실행 비용
- 메모리 사용량 증가

### ❌ **의존성 추가**
- 새로운 라이브러리 의존
- 버전 관리 복잡성

## 🧪 실험 결과 (예상)

### **성능 비교**
```
기존 방식: ~200ms
LangGraph: ~250ms (+25% 오버헤드)
```

### **코드 가독성**
- **기존**: 직관적, 간단
- **LangGraph**: 구조화, 복잡

### **유지보수성**
- **기존**: 단순한 수정
- **LangGraph**: 모듈별 수정 가능

## 🔮 결론 및 권장사항

### **현재 시스템에서는:**
- ✅ **기존 orchestrator 유지** 추천
- ❌ LangGraph 전환 불필요

### **미래에 고려할 상황:**
- 🔄 **복잡한 멀티스텝 워크플로우** 필요시
- 🤖 **대화형 에이전트** 구축시
- 🧠 **상태 유지** 기능 필요시
- 🔀 **동적 분기 로직** 복잡해질 때

### **실험의 가치:**
- 📚 **학습 경험** - 새로운 패러다임 이해
- 🔧 **기술 준비** - 미래 요구사항 대비
- 🎯 **아키텍처 인사이트** - 설계 철학 비교

## 📝 사용 예시

```python
# 직접 사용 (테스트용)
from src.experimental.langgraph_rag import get_langgraph_workflow

workflow = get_langgraph_workflow()
result = workflow.run_workflow("KB 스마트론 금리가 어떻게 되나요?")

print(f"응답: {result['response']}")
print(f"카테고리: {result['category']}")
print(f"소스 수: {len(result['sources'])}")
```

---

**💡 TIP**: 이 실험을 통해 LangGraph의 장단점을 직접 경험하고, 향후 프로젝트에서 적절한 기술 선택을 할 수 있습니다!
