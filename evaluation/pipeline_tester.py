#!/usr/bin/env python3
"""
RAG 파이프라인 테스트 도구
- 다양한 RAG 파이프라인 비교 (LLM, Intent, RAG, LangGraph)
- 응답 시간 및 소스 문서 추적
- 개발 및 디버깅용
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

class PipelineTester:
    def __init__(self, endpoint_type: str = "intent"):
        self.base_url = BASE_URL
        self.results = []
        self.endpoint_type = endpoint_type
        self.endpoint_map = {
            "intent": "/process_with_intent_routing",
            "rag": "/query_rag"
        }
    
    def _print_source_details(self, index: int, source: Dict[str, Any]):
        """소스 문서 상세 정보를 출력하는 헬퍼 메서드"""
        if isinstance(source, dict):
            if 'metadata' in source:
                metadata = source.get("metadata", {})
                page_content = source.get('page_content', '')
            else:
                metadata = source
                page_content = ''
        else:
            metadata = {}
            page_content = ''
        
        file_name = metadata.get('file_name', 'Unknown')
        main_category = metadata.get('main_category', 'Unknown')
        sub_category = metadata.get('sub_category', 'Unknown')
        page_number = metadata.get('page_number', 'Unknown')
        chunk_index = metadata.get('chunk_index', 'Unknown')
        
        print(f"  {index}. 파일명: {file_name}")
        print(f"     카테고리: {main_category}/{sub_category}")
        print(f"     청크: {chunk_index}, 페이지: {page_number}")
        if len(page_content) > 0:
            print(f"     내용: {page_content[:100]}...")

    def check_server(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/healthcheck", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def save_results(self, filename: str):
        """결과를 JSON 파일로 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"결과가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"결과 저장 실패: {e}")

    def test_llm_only(self, prompt: str) -> Dict[str, Any]:
        """LLM 전용 테스트"""
        print(f"\nLLM 전용 테스트")
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
                    "answer": data.get("response", "")
                }
                
                print(f"응답 시간: {result['response_time']}초")
                print(f"답변: {result['answer'][:100]}...")
                print("RAG 없이 LLM만 사용 (빠른 응답)")
                
                return result
            else:
                print(f"오류: {response.status_code}")
                return {"type": "LLM_ONLY", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"예외: {e}")
            return {"type": "LLM_ONLY", "error": str(e)}

    def test_rag_pipeline(self, prompt: str) -> Dict[str, Any]:
        """RAG 파이프라인 테스트"""
        endpoint = self.endpoint_map.get(self.endpoint_type, self.endpoint_map["intent"])
        print(f"\nRAG 테스트 ({self.endpoint_type})")
        print(f"엔드포인트: {endpoint}")
        print(f"질문: {prompt}")

        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json={"prompt": prompt},
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                sources = data.get("sources", [])
                category = data.get("category", "Unknown")
                
                result = {
                    "type": f"RAG_{self.endpoint_type.upper()}",
                    "question": prompt,
                    "response_time": round(end_time - start_time, 2),
                    "status": data.get("status"),
                    "answer": data.get("response", ""),
                    "category": category,
                    "sources_count": len(sources),
                    "sources": sources
                }
                
                print(f"응답 시간: {result['response_time']}초")
                print(f"분류된 카테고리: {category}")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                
                if self.endpoint_type == "intent":
                    if category == "company_products":
                        print("처리 흐름: Intent 분류 → 상품명 추출 → 스마트 검색 (파일명 → 키워드 → 폴더)")
                    else:
                        print(f"처리 흐름: Intent 분류 → {category} 카테고리별 검색")
                else:
                    print("처리 흐름: LLM 기반 상품명 추출 → 키워드 검색 (폴백: 일반 검색)")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"오류: {response.status_code}")
                return {"type": f"RAG_{self.endpoint_type.upper()}", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"예외: {e}")
            return {"type": f"RAG_{self.endpoint_type.upper()}", "error": str(e)}

    def test_basic_rag(self, prompt: str) -> Dict[str, Any]:
        """기본 RAG 테스트"""
        print(f"\n기본 RAG 테스트")
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
                
                print(f"응답 시간: {result['response_time']}초")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                print("처리 흐름: LLM 기반 상품명 추출 → 키워드 검색 (폴백: 일반 검색)")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"오류: {response.status_code}")
                return {"type": "BASIC_RAG", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"예외: {e}")
            return {"type": "BASIC_RAG", "error": str(e)}
    
    def test_langgraph_rag(self, prompt: str) -> Dict[str, Any]:
        """LangGraph RAG 테스트"""
        print(f"\nLangGraph RAG 테스트 (실험용)")
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
                
                print(f"워크플로우 타입: {result['workflow_type']}")
                print(f"분류된 카테고리: {result['category']}")
                print(f"응답 시간: {result['response_time']}초")
                print(f"소스 문서: {result['sources_count']}개")
                print(f"답변: {result['answer'][:100]}...")
                print("처리 흐름: 그래프 기반 노드 실행: classify_intent → search_documents → filter_relevance → generate_response")
                
                # 소스 문서 상세 정보 출력
                if sources:
                    print("소스 문서 상세:")
                    for i, source in enumerate(sources[:3], 1):  # 최대 3개만 표시
                        self._print_source_details(i, source)
                
                return result
            else:
                print(f"오류: {response.status_code}")
                return {"type": "LANGGRAPH_RAG", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            print(f"예외: {e}")
            return {"type": "LANGGRAPH_RAG", "error": str(e)}
    
    def run_comprehensive_test(self, questions: List[str]):
        """종합 테스트 실행"""
        print(f"RAG 시스템 파이프라인 테스트 시작")
        print(f"사용 엔드포인트: {self.endpoint_map.get(self.endpoint_type)} ({self.endpoint_type})")
        print("=" * 60)
        
        if not self.check_server():
            print("서버가 실행되지 않았습니다. 먼저 서버를 시작해주세요.")
            return
        
        print("서버 연결 확인됨")
        
        for i, question in enumerate(questions, 1):
            print(f"\n테스트 {i}/{len(questions)}")
            print("-" * 40)
            
            # LangGraph RAG 테스트만 실행 (다른 테스트는 주석 처리됨)
            langgraph_result = self.test_langgraph_rag(question)
            
            # 결과 저장
            test_result = {
                "question": question,
                "langgraph_rag": langgraph_result
            }
            self.results.append(test_result)
            
            # 잠시 대기
            time.sleep(1)
        
        self.print_summary()
    
    def print_summary(self):
        """테스트 결과 요약"""
        print(f"\n{'='*60}")
        print("RAG 파이프라인 테스트 결과 요약")
        print(f"{'='*60}")
        
        for i, result in enumerate(self.results, 1):
            print(f"\n질문 {i}: {result['question'][:50]}...")
            print("-" * 40)
            
            # LangGraph RAG
            if 'langgraph_rag' in result:
                langgraph = result['langgraph_rag']
                if 'error' not in langgraph:
                    category = langgraph.get('category', 'Unknown')
                    sources_count = langgraph.get('sources_count', 0)
                    print(f"LangGraph RAG: {langgraph['response_time']}초 "
                          f"({category}, 소스: {sources_count}개, 그래프 워크플로우)")
                else:
                    print(f"LangGraph RAG: 오류 - {langgraph['error']}")

def main():
    parser = argparse.ArgumentParser(description="RAG 시스템 파이프라인 테스트 도구")
    parser.add_argument("--question", "-q", help="테스트할 질문")
    parser.add_argument("--file", "-f", help="질문 목록이 담긴 파일 경로")
    parser.add_argument("--save", "-s", help="결과 저장 파일명", default="pipeline_test_results.json")
    parser.add_argument("--type", "-t", choices=["llm", "intent_routing", "rag", "langgraph", "all"], 
                       default="all", help="테스트할 파이프라인 타입")
    parser.add_argument("--category", "-c", help="테스트할 카테고리 (company_products, company_rules, industry_policies_and_regulations, general_banking_FAQs)")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard"], help="테스트할 난이도")
    parser.add_argument("--endpoint", "-e", choices=["intent", "rag"], default="intent", 
                       help="사용할 엔드포인트 (intent: process_with_intent_routing, rag: query_rag)")
    args = parser.parse_args()
    
    tester = PipelineTester(args.endpoint)
    
    # 질문 목록 준비
    questions = []
    
    if args.question:
        questions = [args.question]
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {args.file}")
            return
    else:
        # 카테고리/난이도 필터가 있으면 데이터셋에서 선택
        if args.category or args.difficulty:
            from evaluation.test_dataset import dataset
            
            filtered_cases = dataset
            
            if args.category:
                filtered_cases = [case for case in filtered_cases if case.get("category") == args.category]
                print(f"카테고리 필터 적용: {args.category}")
            
            if args.difficulty:
                filtered_cases = [case for case in filtered_cases if case.get("difficulty") == args.difficulty]
                print(f"난이도 필터 적용: {args.difficulty}")
            
            questions = [case["query"] for case in filtered_cases[:10]]  # 최대 10개로 제한
            print(f"필터링된 질문 수: {len(questions)}개")
        else:
            # 기본 테스트 질문들
            questions = [
                "KB 스마트론에 대해 알려주세요",
                "KB 닥터론 금리가 어떻게 되나요?",
                "대출 한도는 얼마까지 가능한가요?"
            ]

    if not questions:
        print("테스트할 질문이 없습니다.")
        return
    
    # 서버 상태 확인
    if not tester.check_server():
        print("서버가 실행되지 않았습니다.")
        print("다음 명령어로 서버를 시작해주세요:")
        print("python run_server.py")
        return
    
    # 테스트 실행
    if args.type == "all":
        tester.run_comprehensive_test(questions)
    else:
        print(f"{args.type.upper()} 파이프라인 테스트 시작")
        print(f"사용 엔드포인트: {tester.endpoint_map.get(args.endpoint)} ({args.endpoint})")
        print("=" * 60)
        
        for i, question in enumerate(questions, 1):
            print(f"\n테스트 {i}/{len(questions)}: {question}")
            print("-" * 40)
            
            if args.type == "llm":
                result = tester.test_llm_only(question)
            elif args.type == "intent_routing":
                result = tester.test_rag_pipeline(question)
            elif args.type == "rag":
                result = tester.test_basic_rag(question)
            elif args.type == "langgraph":
                result = tester.test_langgraph_rag(question)
            
            tester.results.append({"question": question, "result": result})
            time.sleep(1)
    
    # 결과 저장
    if args.save:
        tester.save_results(args.save)

if __name__ == "__main__":
    main()
