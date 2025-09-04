#!/usr/bin/env python3
"""
종합 RAG 성능 평가 시스템 (통합 버전)
- 문서 검색 성능: MRR, MAP, NDCG, Precision, Recall, F1
- 답변 생성 품질: 정확성, 완전성, 관련성
- 빠른 테스트, 기본 테스트, 종합 평가 통합
- 전체 시스템 성능 분석
"""
import requests
import time
import math
import re
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import sys
import os

# 프로젝트 루트 경로를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.rag_test_dataset import dataset, get_dataset_stats

BASE_URL = "http://localhost:8000/api/v1"

class RAGMetrics:
    """RAG 평가 지표 계산 클래스"""
    
    @staticmethod
    def calculate_precision_at_k(retrieved_files: List[str], relevant_files: List[str], k: int = 5) -> float:
        """Precision@K 계산"""
        if not retrieved_files or k == 0:
            return 0.0
        
        retrieved_k = retrieved_files[:k]
        relevant_retrieved = sum(1 for file in retrieved_k if any(rel in file for rel in relevant_files))
        return relevant_retrieved / len(retrieved_k)
    
    @staticmethod
    def calculate_recall_at_k(retrieved_files: List[str], relevant_files: List[str], k: int = 5) -> float:
        """Recall@K 계산"""
        if not relevant_files:
            return 0.0
        
        retrieved_k = retrieved_files[:k]
        relevant_retrieved = sum(1 for rel_file in relevant_files 
                               if any(rel_file in ret_file for ret_file in retrieved_k))
        return relevant_retrieved / len(relevant_files)
    
    @staticmethod
    def calculate_f1_at_k(retrieved_files: List[str], relevant_files: List[str], k: int = 5) -> float:
        """F1@K 계산"""
        precision = RAGMetrics.calculate_precision_at_k(retrieved_files, relevant_files, k)
        recall = RAGMetrics.calculate_recall_at_k(retrieved_files, relevant_files, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_mrr(retrieved_files: List[str], relevant_files: List[str]) -> float:
        """Mean Reciprocal Rank 계산"""
        for i, retrieved_file in enumerate(retrieved_files, 1):
            if any(rel_file in retrieved_file for rel_file in relevant_files):
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def calculate_ap(retrieved_files: List[str], relevant_files: List[str]) -> float:
        """Average Precision 계산"""
        if not relevant_files:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, retrieved_file in enumerate(retrieved_files, 1):
            if any(rel_file in retrieved_file for rel_file in relevant_files):
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_files) if relevant_files else 0.0
    
    @staticmethod
    def calculate_ndcg_at_k(retrieved_files: List[str], relevant_files: List[str], k: int = 5) -> float:
        """NDCG@K 계산"""
        if not relevant_files or k == 0:
            return 0.0
        
        # DCG 계산
        dcg = 0.0
        for i, retrieved_file in enumerate(retrieved_files[:k], 1):
            relevance = 1 if any(rel_file in retrieved_file for rel_file in relevant_files) else 0
            dcg += relevance / math.log2(i + 1)
        
        # IDCG 계산 (이상적인 순서)
        ideal_relevances = [1] * min(len(relevant_files), k) + [0] * max(0, k - len(relevant_files))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        return dcg / idcg if idcg > 0 else 0.0

class AnswerQualityEvaluator:
    """답변 품질 평가 클래스"""
    
    @staticmethod
    def calculate_keyword_overlap(generated_answer: str, expected_answer: str) -> float:
        """키워드 겹침 비율 계산"""
        if not generated_answer or not expected_answer:
            return 0.0
        
        # 한국어 키워드 추출
        generated_keywords = set(re.findall(r'[가-힣]{2,}', generated_answer.lower()))
        expected_keywords = set(re.findall(r'[가-힣]{2,}', expected_answer.lower()))
        
        if not expected_keywords:
            return 0.0
        
        overlap = len(generated_keywords.intersection(expected_keywords))
        return overlap / len(expected_keywords)
    
    @staticmethod
    def calculate_semantic_similarity(generated_answer: str, expected_answer: str) -> float:
        """의미적 유사도 계산 (간단한 버전)"""
        if not generated_answer or not expected_answer:
            return 0.0
        
        # 핵심 개념 추출
        generated_concepts = set(re.findall(r'[가-힣]{3,}', generated_answer.lower()))
        expected_concepts = set(re.findall(r'[가-힣]{3,}', expected_answer.lower()))
        
        if not expected_concepts:
            return 0.0
        
        # Jaccard 유사도
        intersection = len(generated_concepts.intersection(expected_concepts))
        union = len(generated_concepts.union(expected_concepts))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_completeness(generated_answer: str, expected_answer: str) -> float:
        """답변 완전성 계산"""
        if not generated_answer or not expected_answer:
            return 0.0
        
        # 예상 답변의 핵심 구성 요소들이 포함되어 있는지 확인
        expected_parts = expected_answer.split(',')
        covered_parts = 0
        
        for part in expected_parts:
            part_keywords = re.findall(r'[가-힣]{2,}', part.strip())
            if any(keyword in generated_answer for keyword in part_keywords):
                covered_parts += 1
        
        return covered_parts / len(expected_parts) if expected_parts else 0.0
    
    @staticmethod
    def calculate_relevance(generated_answer: str, query: str) -> float:
        """질문-답변 관련성 계산"""
        if not generated_answer or not query:
            return 0.0
        
        query_keywords = set(re.findall(r'[가-힣]{2,}', query.lower()))
        answer_keywords = set(re.findall(r'[가-힣]{2,}', generated_answer.lower()))
        
        if not query_keywords:
            return 0.0
        
        overlap = len(query_keywords.intersection(answer_keywords))
        return overlap / len(query_keywords)

class QuickTester:
    """빠른 테스트 클래스 (기존 quick_test.py 기능)"""
    
    def __init__(self):
        self.base_url = BASE_URL
    
    def test_query(self, prompt: str, test_name: str):
        """간단한 질의 테스트"""
        print(f"\n🔍 {test_name}")
        print(f"질의: '{prompt}'")
        print("-" * 50)
        
        start_time = time.time()
        try:
            response = requests.post(f"{self.base_url}/query_rag", 
                                   json={"prompt": prompt},
                                   headers={"Content-Type": "application/json"},
                                   timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 응답시간: {end_time - start_time:.3f}초")
                print(f"📄 검색된 문서 수: {len(data.get('sources', []))}")
                print(f"💬 응답: {data.get('response', '')[:100]}...")
                
                # 소스 정보 출력
                sources = data.get('sources', [])
                if sources:
                    print(f"📁 상위 3개 문서:")
                    for i, source in enumerate(sources[:3], 1):
                        print(f"   {i}. {source.get('file_name', 'Unknown')}")
                        if source.get('file_path'):
                            print(f"      경로: {source.get('file_path')}")
                        if source.get('main_category'):
                            print(f"      메인카테고리: {source.get('main_category')}")
                        if source.get('sub_category'):
                            print(f"      서브카테고리: {source.get('sub_category')}")
                        if source.get('page_number'):
                            print(f"      페이지: {source.get('page_number')}")
                        if source.get('keywords'):
                            print(f"      키워드: {source.get('keywords', [])}")
                        print()
            else:
                print(f"❌ 오류: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    def run_quick_tests(self):
        """빠른 테스트 실행"""
        print("🚀 KB RAG 시스템 빠른 테스트")
        print("=" * 60)
        
        # 서버 상태 확인
        try:
            response = requests.get(f"{self.base_url}/healthcheck")
            if response.status_code == 200:
                print("✅ 서버 연결 성공!")
            else:
                print("❌ 서버 응답 오류")
                return
        except:
            print("❌ 서버에 연결할 수 없습니다.")
            print("💡 서버를 먼저 실행해주세요: python run_server.py --reload")
            return
        
        # 테스트 케이스들
        test_cases = [
            ("신용대출 금리", "신용대출 관련 질의"),
            ("KB 주택담보대출", "주택대출 관련 질의"), 
            ("개인정보보호 정책", "정책 관련 질의"),
            ("금융소비자보호법", "법규 관련 질의")
        ]
        
        for prompt, test_name in test_cases:
            self.test_query(prompt, test_name)
            time.sleep(1)  # 서버 부하 방지
        
        print("\n" + "=" * 60)
        print("🎯 빠른 테스트 완료!")
        print("\n💡 더 자세한 메타데이터를 보려면:")
        print("   curl http://localhost:8000/api/v1/vector_store_stats")

class BasicTester:
    """기본 테스트 클래스 (기존 test_rag_system.py 기능)"""
    
    def __init__(self):
        self.base_url = BASE_URL
    
    def check_server(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/healthcheck", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_basic_search(self, query: str) -> Dict[str, Any]:
        """기본 검색 테스트"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/query_rag",
                json={"prompt": query},
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "result_count": len(data.get("sources", [])),
                    "response": data.get("response", ""),
                    "sources": data.get("sources", [])
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_relevance(self, sources: List[Dict], query: str) -> Dict[str, Any]:
        """검색 결과 관련성 분석"""
        if not sources:
            return {"relevance_score": 0, "category_distribution": {}}
        
        query_keywords = query.lower().split()
        total_score = 0
        category_distribution = {}
        
        for source in sources:
            # 카테고리 분포
            category = source.get("main_category", "unknown")
            category_distribution[category] = category_distribution.get(category, 0) + 1
            
            # 관련성 점수 계산
            filename = source.get("file_name", "").lower()
            keywords = source.get("keywords", [])
            
            score = 0
            # 파일명 매칭
            score += sum(1 for kw in query_keywords if kw in filename)
            # 키워드 매칭  
            score += sum(1 for kw in query_keywords 
                        if any(k.lower().find(kw) >= 0 for k in keywords))
            
            total_score += score
        
        return {
            "relevance_score": total_score / len(sources),
            "category_distribution": category_distribution,
            "total_results": len(sources)
        }
    
    def run_basic_tests(self):
        """기본 테스트 실행"""
        print("🧪 KB RAG 시스템 기본 테스트")
        print("=" * 60)
        
        # 서버 연결 확인
        if not self.check_server():
            print("❌ 서버에 연결할 수 없습니다.")
            return
        
        print("✅ 서버 연결 성공!")
        
        # 테스트 케이스
        test_cases = [
            {"query": "신용대출 금리", "description": "상품 - 신용대출"},
            {"query": "KB 주택담보대출", "description": "상품 - 주택대출"},
            {"query": "개인정보보호 정책", "description": "정책 - 개인정보"},
            {"query": "금융소비자보호법", "description": "법규 - 소비자보호"},
            {"query": "윤리강령", "description": "정책 - 윤리"}
        ]
        
        print(f"\n🔍 {len(test_cases)}개 케이스 테스트 시작\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            desc = test_case["description"]
            
            print(f"[{i}/{len(test_cases)}] {desc}")
            print(f"질의: '{query}'")
            print("-" * 50)
            
            # 기본 검색
            basic_result = self.test_basic_search(query)
            if basic_result["success"]:
                basic_analysis = self.analyze_relevance(basic_result["sources"], query)
                print(f"🔸 기본 검색:")
                print(f"   응답시간: {basic_result['response_time']:.3f}초")
                print(f"   결과 수: {basic_result['result_count']}개")
                print(f"   관련성: {basic_analysis['relevance_score']:.2f}")
                print(f"   카테고리: {basic_analysis['category_distribution']}")
                
                # 문서 출처 정보 출력
                if basic_result['sources']:
                    print(f"   📚 상위 문서:")
                    for i, source in enumerate(basic_result['sources'][:3], 1):
                        print(f"      {i}. {source.get('file_name', 'Unknown')}")
                        if source.get('file_path'):
                            print(f"         경로: {source.get('file_path')}")
                        if source.get('main_category'):
                            print(f"         카테고리: {source.get('main_category')}")
                        if source.get('sub_category'):
                            print(f"         서브카테고리: {source.get('sub_category')}")
            else:
                print(f"🔸 기본 검색 실패: {basic_result['error']}")
            
            results.append(basic_result)
            print()
            time.sleep(1)  # API 부하 방지
        
        # 전체 결과 요약
        self.print_summary(results)
    
    def print_summary(self, results: List[Dict]):
        """결과 요약 출력"""
        print("=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        successful_tests = sum(1 for r in results if r["success"])
        
        print(f"✅ 성공한 테스트: {successful_tests}/{len(results)}")
        
        if successful_tests > 0:
            # 평균 성능 계산
            avg_time = sum(r["response_time"] for r in results if r["success"]) / successful_tests
            print(f"\n⚡ 평균 응답시간: {avg_time:.3f}초")
        
        print(f"\n🎯 테스트 완료!")

class ComprehensiveRAGEvaluator:
    """종합 RAG 평가 시스템"""
    
    def __init__(self):
        self.base_url = BASE_URL
        self.metrics = RAGMetrics()
        self.answer_evaluator = AnswerQualityEvaluator()
        self.results = []
    
    def check_server(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/healthcheck", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_single_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """단일 질의에 대한 종합 평가"""
        test_id = test_case.get("id", "unknown")
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]
        expected_files = test_case.get("expected_file", [])
        
        if isinstance(expected_files, str):
            expected_files = [expected_files]
        
        print(f"🔍 평가 중: {test_id}")
        print(f"질의: {query}")
        
        try:
            # RAG 시스템에 질의
            response = requests.post(
                f"{self.base_url}/query_rag",
                json={"prompt": query},
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    "test_id": test_id,
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
            
            data = response.json()
            generated_answer = data.get("response", "")
            sources = data.get("sources", [])
            
            # 검색된 파일명 추출
            retrieved_files = [source.get("file_name", "") for source in sources]
            
            # === 문서 검색 성능 평가 ===
            retrieval_metrics = {
                "precision_at_3": self.metrics.calculate_precision_at_k(retrieved_files, expected_files, 3),
                "precision_at_5": self.metrics.calculate_precision_at_k(retrieved_files, expected_files, 5),
                "recall_at_3": self.metrics.calculate_recall_at_k(retrieved_files, expected_files, 3),
                "recall_at_5": self.metrics.calculate_recall_at_k(retrieved_files, expected_files, 5),
                "f1_at_3": self.metrics.calculate_f1_at_k(retrieved_files, expected_files, 3),
                "f1_at_5": self.metrics.calculate_f1_at_k(retrieved_files, expected_files, 5),
                "mrr": self.metrics.calculate_mrr(retrieved_files, expected_files),
                "map": self.metrics.calculate_ap(retrieved_files, expected_files),
                "ndcg_at_3": self.metrics.calculate_ndcg_at_k(retrieved_files, expected_files, 3),
                "ndcg_at_5": self.metrics.calculate_ndcg_at_k(retrieved_files, expected_files, 5)
            }
            
            # === 답변 품질 평가 ===
            answer_quality = {
                "keyword_overlap": self.answer_evaluator.calculate_keyword_overlap(generated_answer, expected_answer),
                "semantic_similarity": self.answer_evaluator.calculate_semantic_similarity(generated_answer, expected_answer),
                "completeness": self.answer_evaluator.calculate_completeness(generated_answer, expected_answer),
                "relevance": self.answer_evaluator.calculate_relevance(generated_answer, query)
            }
            
            # === 전체 결과 ===
            result = {
                "test_id": test_id,
                "query": query,
                "success": True,
                "generated_answer": generated_answer,
                "expected_answer": expected_answer,
                "retrieved_files": retrieved_files,
                "expected_files": expected_files,
                "retrieval_metrics": retrieval_metrics,
                "answer_quality": answer_quality,
                "sources_count": len(sources)
            }
            
            # 결과 출력
            print(f"📄 검색 성능:")
            print(f"   Precision@5: {retrieval_metrics['precision_at_5']:.3f}")
            print(f"   Recall@5: {retrieval_metrics['recall_at_5']:.3f}")
            print(f"   F1@5: {retrieval_metrics['f1_at_5']:.3f}")
            print(f"   MRR: {retrieval_metrics['mrr']:.3f}")
            print(f"   MAP: {retrieval_metrics['map']:.3f}")
            print(f"   NDCG@5: {retrieval_metrics['ndcg_at_5']:.3f}")
            
            print(f"📚 검색된 문서 출처:")
            if retrieved_files:
                for i, file_info in enumerate(sources[:5], 1):  # 상위 5개만 표시
                    file_name = file_info.get("file_name", "Unknown")
                    file_path = file_info.get("file_path", "")
                    main_category = file_info.get("main_category", "")
                    sub_category = file_info.get("sub_category", "")
                    page_number = file_info.get("page_number", "")
                    
                    print(f"   {i}. {file_name}")
                    if file_path:
                        print(f"      경로: {file_path}")
                    if main_category:
                        print(f"      카테고리: {main_category}")
                    if sub_category:
                        print(f"      서브카테고리: {sub_category}")
                    if page_number:
                        print(f"      페이지: {page_number}")
                    print()
            else:
                print("   검색된 문서 없음")
            
            print(f"💬 답변 품질:")
            print(f"   키워드 겹침: {answer_quality['keyword_overlap']:.3f}")
            print(f"   의미적 유사도: {answer_quality['semantic_similarity']:.3f}")
            print(f"   완전성: {answer_quality['completeness']:.3f}")
            print(f"   관련성: {answer_quality['relevance']:.3f}")
            
            print(f"📋 예상 정답:")
            print(f"   {expected_answer}")
            
            print(f"📁 예상 파일:")
            if expected_files:
                for i, expected_file in enumerate(expected_files, 1):
                    print(f"   {i}. {expected_file}")
            else:
                print("   예상 파일 없음")
            
            print(f"🎯 생성된 답변: {generated_answer[:100]}...")
            
            return result
            
        except Exception as e:
            print(f"❌ 평가 오류: {e}")
            return {
                "test_id": test_id,
                "success": False,
                "error": str(e)
            }
    
    def run_comprehensive_evaluation(self, test_cases: List[Dict[str, Any]], max_tests: int = None):
        """종합 평가 실행"""
        if max_tests:
            test_cases = test_cases[:max_tests]
        
        print(f"🧪 종합 RAG 평가 시작 ({len(test_cases)}개 테스트)")
        print("=" * 80)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            result = self.evaluate_single_query(test_case)
            self.results.append(result)
            print("-" * 60)
            time.sleep(1)  # API 부하 방지
    
    def calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """전체 평가 지표 집계"""
        if not self.results:
            return {}
        
        # HTTP 성공한 테스트
        http_successful_results = [r for r in self.results if r.get("success", False)]
        
        if not http_successful_results:
            return {"error": "No successful evaluations"}
        
        # 실질적으로 유용한 답변을 제공한 테스트 (검색 성능이 일정 수준 이상)
        meaningful_results = []
        no_answer_results = []
        
        for r in http_successful_results:
            # 검색 성능이 낮거나 "해당 정보를 찾을 수 없습니다"인 경우
            if (r["retrieval_metrics"]["precision_at_5"] < 0.1 or 
                r["retrieval_metrics"]["recall_at_5"] < 0.1 or
                "해당 정보를 찾을 수 없습니다" in r.get("generated_answer", "")):
                no_answer_results.append(r)
            else:
                meaningful_results.append(r)
        
        # 검색 성능 평균 (의미있는 결과만)
        retrieval_avg = {}
        if meaningful_results:
            for metric in ["precision_at_3", "precision_at_5", "recall_at_3", "recall_at_5", 
                          "f1_at_3", "f1_at_5", "mrr", "map", "ndcg_at_3", "ndcg_at_5"]:
                values = [r["retrieval_metrics"][metric] for r in meaningful_results]
                retrieval_avg[metric] = sum(values) / len(values)
        else:
            retrieval_avg = {metric: 0.0 for metric in ["precision_at_3", "precision_at_5", "recall_at_3", "recall_at_5", 
                                                        "f1_at_3", "f1_at_5", "mrr", "map", "ndcg_at_3", "ndcg_at_5"]}
        
        # 답변 품질 평균 (의미있는 결과만)
        answer_avg = {}
        if meaningful_results:
            for metric in ["keyword_overlap", "semantic_similarity", "completeness", "relevance"]:
                values = [r["answer_quality"][metric] for r in meaningful_results]
                answer_avg[metric] = sum(values) / len(values)
        else:
            answer_avg = {metric: 0.0 for metric in ["keyword_overlap", "semantic_similarity", "completeness", "relevance"]}
        
        return {
            "total_tests": len(self.results),
            "http_successful_tests": len(http_successful_results),
            "meaningful_tests": len(meaningful_results),
            "no_answer_tests": len(no_answer_results),
            "http_success_rate": len(http_successful_results) / len(self.results),
            "meaningful_success_rate": len(meaningful_results) / len(self.results),
            "retrieval_performance": retrieval_avg,
            "answer_quality": answer_avg
        }
    
    def print_comprehensive_report(self):
        """종합 평가 보고서 출력"""
        print("\n" + "=" * 80)
        print("📊 종합 RAG 성능 평가 보고서")
        print("=" * 80)
        
        aggregate = self.calculate_aggregate_metrics()
        
        if "error" in aggregate:
            print(f"❌ {aggregate['error']}")
            return
        
        print(f"📈 전체 통계:")
        print(f"   총 테스트: {aggregate['total_tests']}개")
        print(f"   HTTP 성공: {aggregate['http_successful_tests']}개")
        print(f"   의미있는 답변: {aggregate['meaningful_tests']}개")
        print(f"   답변 없음: {aggregate['no_answer_tests']}개")
        print(f"   HTTP 성공률: {aggregate['http_success_rate']*100:.1f}%")
        print(f"   의미있는 답변률: {aggregate['meaningful_success_rate']*100:.1f}%")
        
        print(f"\n🔍 문서 검색 성능:")
        retrieval = aggregate['retrieval_performance']
        print(f"   Precision@3: {retrieval['precision_at_3']:.3f}")
        print(f"   Precision@5: {retrieval['precision_at_5']:.3f}")
        print(f"   Recall@3: {retrieval['recall_at_3']:.3f}")
        print(f"   Recall@5: {retrieval['recall_at_5']:.3f}")
        print(f"   F1@3: {retrieval['f1_at_3']:.3f}")
        print(f"   F1@5: {retrieval['f1_at_5']:.3f}")
        print(f"   MRR: {retrieval['mrr']:.3f}")
        print(f"   MAP: {retrieval['map']:.3f}")
        print(f"   NDCG@3: {retrieval['ndcg_at_3']:.3f}")
        print(f"   NDCG@5: {retrieval['ndcg_at_5']:.3f}")
        
        print(f"\n💬 답변 생성 품질:")
        answer = aggregate['answer_quality']
        print(f"   키워드 겹침: {answer['keyword_overlap']:.3f}")
        print(f"   의미적 유사도: {answer['semantic_similarity']:.3f}")
        print(f"   완전성: {answer['completeness']:.3f}")
        print(f"   관련성: {answer['relevance']:.3f}")
        
        # 성능 등급 평가
        overall_retrieval = (retrieval['precision_at_5'] + retrieval['recall_at_5'] + retrieval['f1_at_5']) / 3
        overall_answer = (answer['keyword_overlap'] + answer['semantic_similarity'] + answer['completeness'] + answer['relevance']) / 4
        
        print(f"\n🎯 종합 성능 등급:")
        print(f"   검색 성능: {self._get_performance_grade(overall_retrieval)} ({overall_retrieval:.3f})")
        print(f"   답변 품질: {self._get_performance_grade(overall_answer)} ({overall_answer:.3f})")
        
        # 개선 권장사항
        print(f"\n💡 개선 권장사항:")
        if overall_retrieval < 0.5:
            print("   - 검색 성능이 낮습니다. 메타데이터 필터링이나 임베딩 모델 개선을 고려하세요.")
        if overall_answer < 0.5:
            print("   - 답변 품질이 낮습니다. 프롬프트 엔지니어링이나 LLM 모델 개선을 고려하세요.")
        if retrieval['recall_at_5'] < 0.6:
            print("   - Recall이 낮습니다. 검색 범위를 늘리거나 키워드 매칭을 개선하세요.")
        if answer['completeness'] < 0.6:
            print("   - 답변 완전성이 부족합니다. 더 많은 컨텍스트를 제공하세요.")
    
    def _get_performance_grade(self, score: float) -> str:
        """성능 점수를 등급으로 변환"""
        if score >= 0.8:
            return "🌟 우수"
        elif score >= 0.6:
            return "✅ 양호"
        elif score >= 0.4:
            return "⚠️ 보통"
        else:
            return "❌ 개선필요"

def main():
    print("🧪 KB RAG 시스템 통합 테스트 도구")
    print("=" * 80)
    
    # 데이터셋 통계
    stats = get_dataset_stats()
    print(f"📊 평가 데이터셋:")
    print(f"   총 테스트 케이스: {stats['total_cases']}개")
    print(f"   난이도별: {stats['by_difficulty']}")
    print(f"   카테고리별: {stats['by_subcategory']}")
    
    # 서버 연결 확인
    try:
        response = requests.get(f"{BASE_URL}/healthcheck", timeout=5)
        if response.status_code != 200:
            print("\n❌ 서버에 연결할 수 없습니다.")
            print("💡 서버를 먼저 실행해주세요: python run_server.py --reload")
            return
    except:
        print("\n❌ 서버에 연결할 수 없습니다.")
        print("💡 서버를 먼저 실행해주세요: python run_server.py --reload")
        return
    
    print("\n✅ 서버 연결 성공!")
    
    # 테스트 옵션 선택
    print("\n🎮 테스트 옵션:")
    print("1. 🚀 빠른 테스트 (4개 기본 케이스)")
    print("2. 🧪 기본 테스트 (5개 상세 케이스)")
    print("3. 📊 종합 평가 (데이터셋 기반)")
    print("4. 🔍 빠른 질의 테스트 (직접 입력)")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        # 빠른 테스트
        quick_tester = QuickTester()
        quick_tester.run_quick_tests()
        
    elif choice == "2":
        # 기본 테스트
        basic_tester = BasicTester()
        basic_tester.run_basic_tests()
        
    elif choice == "3":
        # 종합 평가
        evaluator = ComprehensiveRAGEvaluator()
        
        # 평가 옵션 선택
        print("\n📊 평가 옵션:")
        print("1. 빠른 평가 (5개 케이스)")
        print("2. 표준 평가 (10개 케이스)")
        print("3. 전체 평가 (모든 케이스)")
        print("4. 난이도별 평가")
        
        eval_choice = input("\n선택 (1-4): ").strip()
        
        if eval_choice == "1":
            evaluator.run_comprehensive_evaluation(dataset[:5])
        elif eval_choice == "2":
            evaluator.run_comprehensive_evaluation(dataset[:10])
        elif eval_choice == "3":
            confirm = input("⚠️ 전체 평가는 시간이 오래 걸립니다. 계속하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                evaluator.run_comprehensive_evaluation(dataset)
            else:
                print("평가가 취소되었습니다.")
                return
        elif eval_choice == "4":
            difficulty = input("난이도 선택 (easy/medium/hard): ").strip()
            test_cases = [case for case in dataset if case.get("difficulty") == difficulty]
            if test_cases:
                evaluator.run_comprehensive_evaluation(test_cases)
            else:
                print(f"❌ '{difficulty}' 난이도의 테스트 케이스가 없습니다.")
                return
        else:
            print("❌ 잘못된 선택입니다.")
            return
        
        # 종합 보고서 출력
        evaluator.print_comprehensive_report()
        
    elif choice == "4":
        # 빠른 질의 테스트
        print("\n🔍 빠른 질의 테스트")
        print("질문을 입력하면 즉시 RAG 시스템에서 답변을 받을 수 있습니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        
        while True:
            query = input("\n질문: ").strip()
            if query.lower() in ['quit', 'exit', '종료']:
                break
            
            if not query:
                continue
            
            try:
                response = requests.post(
                    f"{BASE_URL}/query_rag",
                    json={"prompt": query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"\n💬 답변: {data.get('response', '')}")
                    
                    sources = data.get('sources', [])
                    if sources:
                        print(f"\n📚 참고 문서:")
                        for i, source in enumerate(sources[:3], 1):
                            print(f"   {i}. {source.get('file_name', 'Unknown')}")
                            if source.get('file_path'):
                                print(f"      경로: {source.get('file_path')}")
                else:
                    print(f"❌ 오류: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
    
    else:
        print("❌ 잘못된 선택입니다.")
        return
    
    print(f"\n🎉 테스트 완료!")

if __name__ == "__main__":
    main()
