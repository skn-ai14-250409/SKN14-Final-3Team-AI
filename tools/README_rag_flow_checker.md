# 🔍 RAG Flow Checker

RAG 시스템의 다양한 파이프라인을 테스트하고 성능을 비교하는 도구입니다.

## 🚀 지원하는 파이프라인

1. **LLM 전용** (`llm`) - RAG 없이 LLM만 사용
2. **Intent 라우팅** (`intent_routing`) - 카테고리 분류 후 처리  
3. **기본 RAG** (`rag`) - 기존 벡터 검색 기반
4. **🧪 LangGraph RAG** (`langgraph`) - 실험용 그래프 워크플로우
5. **종합 테스트** (`all`) - 모든 파이프라인 비교

## 📋 사용법

### 1. 종합 테스트 (모든 파이프라인 비교)
```bash
python tools/rag_flow_checker.py
```

### 2. 특정 파이프라인만 테스트
```bash
# 기본 RAG만 테스트
python tools/rag_flow_checker.py --type rag

# 🧪 LangGraph 실험용 테스트
python tools/rag_flow_checker.py --type langgraph

# Intent 라우팅 테스트
python tools/rag_flow_checker.py --type intent_routing

# LLM 전용 테스트
python tools/rag_flow_checker.py --type llm
```

### 3. 특정 질문으로 테스트
```bash
python tools/rag_flow_checker.py --question "KB 스마트론에 대해 알려주세요" --type langgraph
```

### 4. 질문 파일로 배치 테스트
```bash
# questions.txt 파일 생성
echo "KB 스마트론 금리가 어떻게 되나요?" > questions.txt
echo "대출 한도는 얼마까지 가능한가요?" >> questions.txt

# 배치 테스트 실행
python tools/rag_flow_checker.py --file questions.txt --type langgraph
```

## 📊 출력 예시

### LangGraph RAG 테스트 결과:
```
🧪 LangGraph RAG 테스트 (실험용)
질문: KB 스마트론에 대해 알려주세요

🔬 워크플로우 타입: langgraph
📂 분류된 카테고리: company_products
✅ 응답 시간: 1.23초
소스 문서: 3개
답변: KB 스마트론은 개인신용대출 상품으로...
💡 그래프 기반 노드 실행: classify_intent → search_documents → filter_relevance → generate_response

📄 소스 문서 상세:
1. 파일명: KB_스마트론.pdf
   카테고리: 상품/개인 신용대출
   청크: 2/15, 페이지: 1
```

### 종합 테스트 비교 결과:
```
📝 테스트 1/3: KB 스마트론에 대해 알려주세요
----------------------------------------

🤖 LLM 전용: 0.8초
🚀 Intent 라우팅: 1.1초 (company_products, 소스: 4개, 스마트 필터링)
🔍 기본 RAG: 1.0초
🧪 LangGraph RAG: 1.2초 (company_products, 소스: 3개, 그래프 워크플로우)
```

## 🔧 옵션

- `--question, -q`: 테스트할 단일 질문
- `--file, -f`: 질문 목록 파일 경로
- `--type, -t`: 파이프라인 타입 (`llm`, `intent_routing`, `rag`, `langgraph`, `all`)
- `--save, -s`: 결과 저장 파일명 (기본: `rag_flow_test_results.json`)

## 🧪 LangGraph vs 기본 RAG 비교

### **LangGraph의 특징:**
- ✅ **그래프 기반 워크플로우** - 노드 단위 실행 추적
- ✅ **중앙화된 상태 관리** - 디버깅 용이
- ✅ **모듈형 구조** - 각 단계가 독립적
- ✅ **조건부 라우팅** - 복잡한 분기 로직

### **성능 비교 (예상):**
```
기본 RAG:     ~1.0초 (단순한 실행 흐름)
LangGraph:    ~1.2초 (그래프 오버헤드 +20%)
```

### **언제 LangGraph를 고려할까:**
- 🔄 복잡한 멀티스텝 워크플로우 필요
- 🤖 대화형 에이전트 구축
- 🧠 상태 유지 기능 필요
- 🔀 동적 분기 로직이 복잡할 때

## 💡 활용 팁

### 1. 성능 비교
```bash
# 동일한 질문으로 두 방식 비교
python tools/rag_flow_checker.py --question "KB 스마트론 금리는?" --type rag
python tools/rag_flow_checker.py --question "KB 스마트론 금리는?" --type langgraph
```

### 2. 배치 성능 테스트
```bash
# 여러 질문으로 종합 성능 비교
python tools/rag_flow_checker.py --file test_questions.txt --type all
```

### 3. 결과 저장 및 분석
```bash
# 결과를 JSON 파일로 저장
python tools/rag_flow_checker.py --save langgraph_test_results.json --type langgraph

# 저장된 결과 확인
cat langgraph_test_results.json | jq '.[] | {question: .question, response_time: .result.response_time}'
```

## 🔍 트러블슈팅

### 1. 서버 연결 실패
```bash
# 서버 시작 확인
python run_server.py

# 서버 상태 확인
curl http://localhost:8000/api/v1/healthcheck
```

### 2. LangGraph 의존성 오류
```bash
# LangGraph 설치
pip install langgraph==0.2.67

# 의존성 재설치
pip install -r requirements.txt
```

### 3. 타임아웃 오류
```bash
# 타임아웃 시간 증가 (코드 수정 필요)
# tools/rag_flow_checker.py 파일에서 timeout=300 값 증가
```

---

**💡 TIP**: LangGraph는 실험용이므로 기존 시스템과 성능을 비교하며 학습 목적으로 활용하세요!