#!/usr/bin/env python3
"""
RAG 시스템 호출 흐름 체크 도구
- 어떤 RAG 파이프라인이 호출되는지 확인
- 라우팅 결과 및 소스 문서 추적
- 성능 및 응답 시간 측정
"""
import requests
import time
import json
import argparse
from typing import Dict, Any, List
from pathlib import Path
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_URL = "http://localhost:8000/api/v1"

class RAGFlowChecker:
    def __init__(self):
        self.base_url = BASE_URL
        self.results = []
    
    def _print_source_details(self, index: int, source: Dict[str, Any]):
        """소스 문서 상세 정보를 출력하는 헬퍼 메서드"""
        # source는 이미 metadata 딕셔너리이거나 Document 객체의 metadata일 수 있음
        if isinstance(source, dict):
            if 'metadata' in source:
                # Document 객체 형태
                metadata = source.get("metadata", {})
                page_content = source.get('page_content', '')
            else:
                # 이미 metadata 딕셔너리
                metadata = source
                page_content = ''
        else:
            metadata = {}
            page_content = ''
        
        file_name = metadata.get('file_name', 'Unknown')
        main_category = metadata.get('main_category', 'Unknown')
        sub_category = metadata.get('sub_category', 'Unknown')
        file_path = metadata.get('file_path', 'Unknown')
        page_number = metadata.get('page_number', 'Unknown')
        chunk_index = metadata.get('chunk_index', 'Unknown')
        
        print(f"  {index}. 📄 {file_name}")
        print(f"     📁 경로: {file_path}")
        print(f"     🏷️  카테고리: {main_category} → {sub_category}")
        print(f"     📄 페이지: {page_number}, 청크: {chunk_index}")
        if page_content:
            content_preview = page_content[:100].replace('\n', ' ')
            print(f"     📝 내용 미리보기: {content_preview}...")
        print()
    
    def check_server(self):
        """서버 상태 확인"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_llm_only(self, prompt: str) -> Dict[str, Any]:
        """LLM 전용 답변 테스트"""
        print(f"\n🔍 LLM 전용 답변 테스트")
        print(f"질문: {prompt}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/answer_with_llm_only",
                json={"prompt": prompt},
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    "type": "LLM_ONLY",
                    "question": prompt,
                    "response_time": round(end_time - start_time, 2),
                    "status": data.get("status"),
                    "answer": data.get("response", ""),
                    "sources": []
                }
                print(f"✅ 응답 시간: {result['response_time']}초")
                print(f"답변: {result['answer'][:100]}...")
                return result
            else:
                print(f"❌ 오류: {response.status_code}")
                return {"type": "LLM_ONLY", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ 예외: {e}")
            return {"type": "LLM_ONLY", "error": str(e)}
    
    
    def test_intent_routing(self, prompt: str) -> Dict[str, Any]:
        """기존 제임스 Intent 라우팅 테스트"""
        print(f"\n🚀 Intent 라우팅 테스트")
        print(f"질문: {prompt}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/process_with_intent_routing",
                json={"prompt": prompt},
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                category = data.get("category", "")
                
                result = {
                    "type": "INTENT_ROUTING",
                    "question": prompt,
                    "response_time": round(end_time - start_time, 2),
                    "status": data.get("status"),
                    "answer": data.get("response", ""),
                    "category": category,
                    "sources_count": len(sources),
                    "sources": sources
                }
                
                print(f"✅ 응답 시간: {result['response_time']}초")
                print(f"분류: {category}")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                
                # 카테고리별 검색 정보 표시
                if category == "company_products":
                    print("🔍 상품 폴더로 검색됨 (MAIN_PRODUCT)")
                    print("💡 LLM 기반 상품명 추출 → 키워드 검색")
                    print("📋 추출된 상품명과 키워드는 서버 로그에서 확인 가능")
                elif category == "company_rules":
                    print("🔍 내규 폴더로 검색됨 (MAIN_RULE)")
                    print("📋 similarity_search_by_folder 사용")
                elif category == "industry_policies_and_regulations":
                    print("🔍 법률 폴더로 검색됨 (MAIN_LAW)")
                    print("📋 similarity_search_by_folder 사용")
                elif category == "general_banking_FAQs":
                    print("🔍 일반 FAQ - LLM 전용 답변")
                    print("📋 RAG 없이 LLM만 사용")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("📄 소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"❌ 오류: {response.status_code}")
                return {"type": "INTENT_ROUTING", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ 예외: {e}")
            return {"type": "INTENT_ROUTING", "error": str(e)}

    def test_basic_rag(self, prompt: str) -> Dict[str, Any]:
        """기본 RAG 테스트"""
        print(f"\n🔍 기본 RAG 테스트")
        print(f"질문: {prompt}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/query_rag",
                json={"prompt": prompt},
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                
                result = {
                    "type": "BASIC_RAG",
                    "question": prompt,
                    "response_time": round(end_time - start_time, 2),
                    "status": data.get("status"),
                    "answer": data.get("response", ""),
                    "sources_count": len(sources),
                    "sources": sources
                }
                
                print(f"✅ 응답 시간: {result['response_time']}초")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                print("💡 LLM 기반 상품명 추출 → 키워드 검색 (폴백: 일반 검색)")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("📄 소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"❌ 오류: {response.status_code}")
                return {"type": "BASIC_RAG", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ 예외: {e}")
            return {"type": "BASIC_RAG", "error": str(e)}
    
    def test_langgraph_rag(self, prompt: str) -> Dict[str, Any]:
        """🧪 실험용 LangGraph RAG 테스트"""
        print(f"\n🧪 LangGraph RAG 테스트 (실험용)")
        print(f"질문: {prompt}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/experimental/langgraph_rag",
                json={"prompt": prompt},
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                
                result = {
                    "type": "LANGGRAPH_RAG",
                    "question": prompt,
                    "response_time": round(end_time - start_time, 2),
                    "status": data.get("status"),
                    "answer": data.get("response", ""),
                    "sources_count": len(sources),
                    "sources": sources,
                    "category": data.get("category", "unknown"),
                    "workflow_type": data.get("workflow_type", "unknown"),
                    "experimental": data.get("experimental", False)
                }
                
                print(f"🔬 워크플로우 타입: {result['workflow_type']}")
                print(f"📂 분류된 카테고리: {result['category']}")
                print(f"✅ 응답 시간: {result['response_time']}초")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                print("💡 그래프 기반 노드 실행: classify_intent → search_documents → filter_relevance → generate_response")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("📄 소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"❌ 오류: {response.status_code}")
                return {"type": "LANGGRAPH_RAG", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"❌ 예외: {e}")
            return {"type": "LANGGRAPH_RAG", "error": str(e)}
    
    def run_comprehensive_test(self, questions: List[str]):
        """종합 테스트 실행"""
        print(f"🚀 RAG 시스템 종합 테스트 시작")
        print("=" * 60)
        
        if not self.check_server():
            print("❌ 서버가 실행되지 않았습니다. 먼저 서버를 시작해주세요.")
            return
        
        print("✅ 서버 연결 확인됨")
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 테스트 {i}/{len(questions)}")
            print("-" * 40)
            
            # # 1. LLM 전용 테스트
            # llm_result = self.test_llm_only(question)
            
            # # 2. Intent 라우팅 테스트
            # intent_routing_result = self.test_intent_routing(question)
            
            # # 3. 기본 RAG 테스트
            # rag_result = self.test_basic_rag(question)
            
            # 4. 🧪 LangGraph RAG 테스트
            langgraph_result = self.test_langgraph_rag(question)
            
            # 결과 저장
            test_result = {
                "question": question,
                # "llm_only": llm_result,
                # "intent_routing": intent_routing_result,
                # "basic_rag": rag_result,
                "langgraph_rag": langgraph_result
            }
            self.results.append(test_result)
            
            # 잠시 대기
            time.sleep(1)
        
        self.print_summary()
    
    def evaluate_answer(self, actual_answer: str, expected_answer: str) -> Dict[str, Any]:
        """답변 평가 - 키워드 기반 유사도 계산"""
        import re
        
        # 한글 키워드 추출
        actual_keywords = set(re.findall(r'[가-힣]{2,}', actual_answer))
        expected_keywords = set(re.findall(r'[가-힣]{2,}', expected_answer))
        
        # 교집합과 합집합 계산
        intersection = actual_keywords.intersection(expected_keywords)
        union = actual_keywords.union(expected_keywords)
        
        # 유사도 계산 (Jaccard similarity)
        similarity = len(intersection) / len(union) if union else 0
        
        return {
            "similarity": round(similarity, 3),
            "matched_keywords": list(intersection),
            "actual_keywords": list(actual_keywords),
            "expected_keywords": list(expected_keywords),
            "is_correct": similarity >= 0.3  # 30% 이상 유사하면 정답으로 판정
        }

    def check_expected_pdf(self, sources: List[Dict], expected_pdf: Dict) -> bool:
        """예상 PDF 파일이 소스에 포함되어 있는지 확인"""
        expected_filename = expected_pdf["file"]
        
        for source in sources:
            if source.get("file_name") == expected_filename:
                return True
        
        return False

    def print_summary(self):
        """테스트 결과 요약"""
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        intent_correct = 0
        rag_correct = 0
        intent_pdf_correct = 0
        rag_pdf_correct = 0
        
        for i, result in enumerate(self.results, 1):
            print(f"\n질문 {i}: {result['question']}")
            print("-" * 40)
            
            # LLM 전용
            llm = result['llm_only']
            if 'error' not in llm:
                print(f"🤖 LLM 전용: {llm['response_time']}초")
            else:
                print(f"🤖 LLM 전용: 오류 - {llm['error']}")
            
            # Intent 라우팅
            intent_routing = result['intent_routing']
            if 'error' not in intent_routing:
                category = intent_routing.get('category', 'Unknown')
                sources_count = intent_routing.get('sources_count', 0)
                if category == "company_products":
                    print(f"🚀 Intent 라우팅: {intent_routing['response_time']}초 "
                          f"({category}, 소스: {sources_count}개, 스마트 필터링)")
                else:
                    print(f"🚀 Intent 라우팅: {intent_routing['response_time']}초 "
                          f"({category}, 소스: {sources_count}개)")
            else:
                print(f"🚀 Intent 라우팅: 오류 - {intent_routing['error']}")
            
            # 예상 답변이 있는 경우 평가
            if i <= len(expected_answers):
                expected_answer = expected_answers[i-1]
                expected_pdf = expected_pdfs[i-1]
                
                print(f"📋 예상 답변: {expected_answer[:50]}...")
                print(f"📄 예상 PDF: {expected_pdf['file']}")
                
                # Intent 라우팅 평가
                intent_routing = result['intent_routing']
                if 'error' not in intent_routing:
                    intent_eval = self.evaluate_answer(intent_routing['answer'], expected_answer)
                    intent_pdf_match = self.check_expected_pdf(intent_routing.get('sources', []), expected_pdf)
                    
                    if intent_eval['is_correct']:
                        intent_correct += 1
                    if intent_pdf_match:
                        intent_pdf_correct += 1
                    
                    print(f"🚀 Intent 라우팅: {intent_routing['response_time']}초 "
                          f"(유사도: {intent_eval['similarity']}, PDF 매치: {'✅' if intent_pdf_match else '❌'})")
                else:
                    print(f"🚀 Intent 라우팅: 오류 - {intent_routing['error']}")
                
                # 기본 RAG 평가
                rag = result['basic_rag']
                if 'error' not in rag:
                    rag_eval = self.evaluate_answer(rag['answer'], expected_answer)
                    rag_pdf_match = self.check_expected_pdf(rag.get('sources', []), expected_pdf)
                    
                    if rag_eval['is_correct']:
                        rag_correct += 1
                    if rag_pdf_match:
                        rag_pdf_correct += 1
                    
                    print(f"🔍 기본 RAG: {rag['response_time']}초 "
                          f"(유사도: {rag_eval['similarity']}, PDF 매치: {'✅' if rag_pdf_match else '❌'})")
                else:
                    print(f"🔍 기본 RAG: 오류 - {rag['error']}")
            else:
                # 예상 답변이 없는 경우 기본 출력
                intent_routing = result['intent_routing']
                if 'error' not in intent_routing:
                    print(f"🚀 Intent 라우팅: {intent_routing['response_time']}초")
                else:
                    print(f"🚀 Intent 라우팅: 오류 - {intent_routing['error']}")
                
                rag = result['basic_rag']
                if 'error' not in rag:
                    print(f"🔍 기본 RAG: {rag['response_time']}초")
                else:
                    print(f"🔍 기본 RAG: 오류 - {rag['error']}")
                
                # 🧪 LangGraph RAG
                if 'langgraph_rag' in result:
                    langgraph = result['langgraph_rag']
                    if 'error' not in langgraph:
                        category = langgraph.get('category', 'Unknown')
                        sources_count = langgraph.get('sources_count', 0)
                        print(f"🧪 LangGraph RAG: {langgraph['response_time']}초 "
                              f"({category}, 소스: {sources_count}개, 그래프 워크플로우)")
                    else:
                        print(f"🧪 LangGraph RAG: 오류 - {langgraph['error']}")
        
        # 전체 정확도 출력
        total_questions = min(len(self.results), len(expected_answers))
        if total_questions > 0:
            print(f"\n{'='*60}")
            print(f"📈 정확도 평가 결과")
            print(f"{'='*60}")
            print(f"🚀 Intent 라우팅 답변 정확도: {intent_correct}/{total_questions} ({intent_correct/total_questions*100:.1f}%)")
            print(f"🚀 Intent 라우팅 PDF 매치: {intent_pdf_correct}/{total_questions} ({intent_pdf_correct/total_questions*100:.1f}%)")
            print(f"🔍 기본 RAG 답변 정확도: {rag_correct}/{total_questions} ({rag_correct/total_questions*100:.1f}%)")
            print(f"🔍 기본 RAG PDF 매치: {rag_pdf_correct}/{total_questions} ({rag_pdf_correct/total_questions*100:.1f}%)")
        
        print(f"\n테스트 완료! 총 {len(self.results)}개 질문 처리")
    
    def save_results(self, filename: str = "rag_flow_test_results.json"):
        """결과를 JSON 파일로 저장"""
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과가 {output_path}에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(description="RAG 시스템 호출 흐름 체크 도구")
    parser.add_argument("--question", "-q", help="테스트할 질문")
    parser.add_argument("--file", "-f", help="질문 목록이 담긴 파일 경로")
    parser.add_argument("--save", "-s", help="결과 저장 파일명", default="rag_flow_test_results.json")
    parser.add_argument("--type", "-t", choices=["llm", "intent_routing", "rag", "langgraph", "all"], 
                       default="all", help="테스트할 파이프라인 타입")
    args = parser.parse_args()
    
    checker = RAGFlowChecker()
    
    # 질문 목록 준비
    questions = []
    
    if args.question:
        questions = [args.question]
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
            return
    else:
        # 기본 테스트 질문들
        questions = [
            # "중도금 문제로 온 고객이 있는데, 중도금 대출 대상에 제약이 있나요?",
            # "회사에서 직원 대출 요청 들어왔는데, 예금 담보로 얼마까지 가능한가요?",
            # "임대주택 입주자 전세자금 특례 보증, 한도 기준이 어떻게 되나요?",
            # "담보대출 채무조정 신청 조건이 뭐였지? 바로 신청 가능한가요?",
            # "오피스텔·상가 중도금 건, 상환 방식은 어떤 옵션들이 있나요?",
           00
        ]

        answers = [
            # "분양주체(국가·지자체·주택공사 등)로부터 계약한 용지에 한해 중도금 대출 대상이 되는 경우가 많습니다.",
            # "법인이 제공한 정기예금(담보)의 일정 비율 내에서 대출이 허용되며, 통상 예금액의 최대 90~95% 이내 규정이 적용됩니다.",
            # "공공임대 관련 특례의 경우 통상 임차보증금의 일정 비율(예: 80~90% 범위) 내에서 한도가 정해지고, 일부 상품은 상한액이 별도로 규정됩니다.",
            # "가계 부동산 담보대출이라면 신규일로부터 일정 기간(예: 1년) 경과, 연체·중대한 부적격 사유가 없을 것 등 기본 자격을 충족해야 신청이 가능합니다.",
            # "일시상환, 원리금균등분할, 원금균등분할 등 여러 방식이 제공되며 상품별로 거치기간·최소·최대 기간 제한이 존재합니다.",
            # "대출 약관에 따라 최초 월상환액을 일정 기간(예: 10년) 고정하는 형태가 있으며, 그 기간 동안 월상환액은 변동되지 않습니다.",
            # "부동산 담보신탁이나 수익증권을 담보로 활용하는 사례가 있지만, 담보 평가·법적 구조에 따라 승인여부와 한도가 달라집니다.",
            "분양토지 중도금은 분양계약 체결 대상(공공·공사 등)이 주로 대상이며, 개인의 경우에는 분양주체와 계약조건에 따라 달라집니다.",
            "대환대출은 기존 채무의 잔액·신용·담보 상태에 따라 승인되며, 일부 상품은 특정 기간 경과 요건이나 잔액 제한 등이 있습니다.",
            "신분증, 등본(주민등록등본), 소득증빙(근로소득원천징수영수증 등), 담보관련 서류(등기부등본, 매매계약서 등)를 우선 수집합니다."
        ]

        pdfs = [
            # "KB_분양토지_중도금대출pdf",
            # "KB_법인_예금담보_임직원.대출.pdf",
            # "KB_임대주택_입주자_특례보증_전세자금대출.pdf",
            # "KB_부동산_담보대출_채무조정_전환제도.pdf",
            # "KB_오피스텔_상가에_대한_중도금(잔금)대출(준주택_포함).pdf",
            # "KB_월상환액_고정형_주택담보대출.pdf",
            # "KB_부동산_담보신탁_수익증권증서_담보대출.pdf",
            "KB_분양토지_중도금대출.pdf",
            "KB_폐업지원_대환대출(신용).pdf",
            "KB_일반부동산_담보대출.pdf"
        ]

    if not questions:
        print("❌ 테스트할 질문이 없습니다.")
        return
    
    # 서버 상태 확인
    if not checker.check_server():
        print("❌ 서버가 실행되지 않았습니다.")
        print("다음 명령어로 서버를 시작해주세요:")
        print("python run_server.py")
        return
    
    # 테스트 실행
    if args.type == "all":
        checker.run_comprehensive_test(questions)
    else:
        print(f"🚀 {args.type.upper()} 파이프라인 테스트 시작")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 테스트 {i}/{len(questions)}: {question}")
            print("-" * 40)
            
            if args.type == "llm":
                result = checker.test_llm_only(question)
            elif args.type == "intent_routing":
                result = checker.test_intent_routing(question)
            elif args.type == "rag":
                result = checker.test_basic_rag(question)
            elif args.type == "langgraph":
                result = checker.test_langgraph_rag(question)
            
            checker.results.append({"question": question, "result": result})
            time.sleep(1)
    
    # 결과 저장
    if args.save:
        checker.save_results(args.save)

if __name__ == "__main__":
    main()
