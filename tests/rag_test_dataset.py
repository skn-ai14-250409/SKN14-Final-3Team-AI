# tests/rag_test_dataset.py
"""
KB금융 창구 행원 실무 RAG 테스트 데이터셋
- 실제 PDF 문서 내용 기반 질문들
- 창구에서 자주 받는 고객 질문들
- 상품 안내 및 업무 관련 질문들
"""

# 창구 행원 실무 질문 데이터셋 (실제 PDF 기반)
dataset = [
    # === 상품 관련 질문 ===
    {
        "id": "product_001",
        "query": "KB 신용대출의 대출 한도는 어떻게 되나요?",
        "expected_answer": "KB 신용대출의 대출 한도는 고객의 소득, 신용도, 담보 등을 종합적으로 고려하여 결정됩니다. 일반적으로 연소득의 3-5배 수준이며, 최대 3억원까지 가능합니다.",
        "expected_file": ["KB_신용대출.pdf"],
        "difficulty": "easy",
        "category": "product",
        "subcategory": "credit_loan"
    },
    {
        "id": "product_002", 
        "query": "KB 햇살론의 특징은 무엇인가요?",
        "expected_answer": "KB 햇살론은 저소득층을 위한 저금리 대출 상품으로, 정부 지원을 받아 일반 신용대출보다 낮은 금리로 대출받을 수 있습니다. 소득 기준과 신용도에 따라 대출 한도가 결정됩니다.",
        "expected_file": ["KB_햇살론15_.pdf"],
        "difficulty": "medium",
        "category": "product", 
        "subcategory": "credit_loan"
    },
    {
        "id": "product_003",
        "query": "KB 급여이체신용대출의 조건은?",
        "expected_answer": "KB 급여이체신용대출은 급여를 KB은행에 이체하는 고객이 이용할 수 있는 대출 상품입니다. 급여이체를 통해 대출금을 상환하며, 일반 신용대출보다 낮은 금리와 높은 한도를 제공합니다.",
        "expected_file": ["KB_급여이체신용대출.pdf"],
        "difficulty": "medium",
        "category": "product",
        "subcategory": "credit_loan"
    },
    
    # === 고객 상담 관련 질문 ===
    {
        "id": "consultation_001",
        "query": "고객이 개인정보 변경을 원할 때 필요한 서류는?",
        "expected_answer": "개인정보 변경 시 신분증(주민등록증, 운전면허증, 여권 중 1개)과 변경사항 증빙서류가 필요합니다. 주소 변경 시 주민등록등본, 연락처 변경 시 본인 확인이 필요합니다.",
        "expected_file": ["KB금융그룹_개인정보보호_정책.pdf"],
        "difficulty": "easy",
        "category": "customer_service",
        "subcategory": "information_change"
    },
    {
        "id": "consultation_002",
        "query": "고객이 계좌 해지를 원할 때 주의사항은?",
        "expected_answer": "계좌 해지 시 잔액 확인, 미해지 자동이체나 결제서비스 해지, 카드 연동 해제, 보안카드 반납 등을 확인해야 합니다. 해지 후에는 재개설이 제한될 수 있습니다.",
        "expected_file": ["내부통제규정.pdf"],
        "difficulty": "easy",
        "category": "customer_service",
        "subcategory": "account_closing"
    },
    {
        "id": "consultation_003",
        "query": "고객이 대출 상담을 원할 때 주의사항은?",
        "expected_answer": "대출 상담 시 고객의 소득과 신용도를 정확히 파악하고, 대출 가능 여부를 명확히 안내해야 합니다. 과도한 대출 권유나 허위 정보 제공은 금지되며, 고객의 상환능력을 고려한 적정한 대출을 권장해야 합니다.",
        "expected_file": ["여신기본강령_및_모범규준.pdf", "윤리강령.pdf"],
        "difficulty": "medium",
        "category": "customer_service",
        "subcategory": "loan_consultation"
    },
    
    # === 법규 준수 관련 질문 ===
    {
        "id": "compliance_001",
        "query": "고객이 의심스러운 거래를 요청할 때 어떻게 해야 하나요?",
        "expected_answer": "의심스러운 거래 요청 시 즉시 거부하고, 해당 거래 내용을 내부신고 시스템에 보고해야 합니다. 고객에게는 관련 법규와 은행 내규를 설명하고, 필요시 금융감독원이나 수사기관에 신고합니다.",
        "expected_file": ["범죄예방관련정책.pdf", "KB금융그룹_내부신고자_보호_정책.pdf"],
        "difficulty": "hard",
        "category": "compliance",
        "subcategory": "suspicious_transaction"
    },
    {
        "id": "compliance_002",
        "query": "고객 정보를 외부에 유출하면 어떻게 되나요?",
        "expected_answer": "고객 정보 외부 유출 시 개인정보보호법 위반으로 최대 3년 이하의 징역 또는 3천만원 이하의 벌금이 부과될 수 있습니다. 은행 내규에 따라 징계처분을 받게 되며, 관련 업무에서 제외됩니다.",
        "expected_file": ["KB금융그룹_개인정보보호_정책.pdf", "임직원_법규준수_행동기준.pdf"],
        "difficulty": "hard",
        "category": "compliance",
        "subcategory": "information_security"
    },
    {
        "id": "compliance_003",
        "query": "임직원이 이해관계자로부터 금품을 받으면 어떻게 되나요?",
        "expected_answer": "임직원이 이해관계자로부터 금품을 받는 것은 엄격히 금지됩니다. 가족을 통한 간접적인 금품 수령도 임직원 본인의 행위로 간주되어 금지되며, 위반 시 징계처분을 받게 됩니다.",
        "expected_file": ["KB금융_반부패_뇌물정책.pdf", "임직원_법규준수_행동기준.pdf"],
        "difficulty": "medium",
        "category": "compliance",
        "subcategory": "bribery_prevention"
    },
    
    # === 창구 업무 프로세스 관련 질문 ===
    {
        "id": "process_001",
        "query": "창구에서 현금 입금 시 주의사항은?",
        "expected_answer": "현금 입금 시 지폐의 진위 여부를 확인하고, 입금표에 금액과 입금자 정보를 정확히 기재해야 합니다. 대량 현금 입금 시 고객 신분증 확인이 필요하며, 의심스러운 경우 내부신고 절차를 따릅니다.",
        "expected_file": ["내부통제규정.pdf", "범죄예방관련정책.pdf"],
        "difficulty": "easy",
        "category": "process",
        "subcategory": "cash_deposit"
    },
    {
        "id": "process_002",
        "query": "창구에서 고객 민원 접수 시 어떻게 해야 하나요?",
        "expected_answer": "고객 민원 접수 시 즉시 담당자에게 연결하거나 민원접수부서로 안내합니다. 민원 내용을 정확히 파악하고, 해결 가능한 범위 내에서 즉시 조치하며, 해결이 어려운 경우 정확한 처리기한을 안내합니다.",
        "expected_file": ["내부통제규정.pdf", "윤리강령.pdf"],
        "difficulty": "easy",
        "category": "process",
        "subcategory": "customer_complaint"
    },
    {
        "id": "process_003",
        "query": "창구에서 고객 상담 시 지켜야 할 윤리적 원칙은?",
        "expected_answer": "창구에서 고객 상담 시 정직, 신뢰, 고객중심의 원칙을 지켜야 합니다. 고객의 권익을 보호하고, 공정하고 투명한 서비스를 제공하며, 개인정보를 철저히 보호해야 합니다.",
        "expected_file": ["윤리강령.pdf", "KB금융그룹_개인정보보호_정책.pdf"],
        "difficulty": "medium",
        "category": "process",
        "subcategory": "ethical_consultation"
    },
    
    # === 특수 상황 대응 관련 질문 ===
    {
        "id": "emergency_001",
        "query": "창구에서 고객이 성희롱이나 괴롭힘을 당했다고 신고할 때 어떻게 해야 하나요?",
        "expected_answer": "성희롱이나 괴롭힘 신고 시 즉시 해당 내용을 내부신고 시스템에 보고하고, 신고자의 신분을 보장해야 합니다. 관련 부서에 즉시 연락하여 적절한 조치를 취하며, 신고자에 대한 불이익 조치를 금지합니다.",
        "expected_file": ["KB금융그룹_직장_내 성희롱·괴롭힘_및_차별_예방_선언서.pdf", "KB금융그룹_내부신고자_보호_정책.pdf"],
        "difficulty": "hard",
        "category": "emergency",
        "subcategory": "harassment_response"
    },
    {
        "id": "emergency_002",
        "query": "창구에서 고객이 안전사고를 당했을 때 대응 방법은?",
        "expected_answer": "창구에서 안전사고 발생 시 즉시 응급처치를 시도하고, 필요시 119에 신고합니다. 사고 현장을 보호하고, 다른 고객들은 안전한 거리로 대피시킵니다. 사고 내용을 즉시 보고하고, 관련 기록을 보존합니다.",
        "expected_file": ["그룹_안전보건_목표 및_경영방침.pdf", "내부통제규정.pdf"],
        "difficulty": "medium",
        "category": "emergency",
        "subcategory": "safety_accident"
    },
    {
        "id": "emergency_003",
        "query": "창구에서 고객이 공정하지 않은 채권회수에 대해 항의할 때 어떻게 해야 하나요?",
        "expected_answer": "공정하지 않은 채권회수 항의 시 즉시 해당 내용을 담당 부서에 보고하고, 고객의 불만사항을 정확히 파악합니다. 공정한 채권회수 정책에 따라 적절한 조치를 취하며, 필요시 상위 부서에 검토를 요청합니다.",
        "expected_file": ["공정한_채권회수를_위한_정책.pdf", "내부통제규정.pdf"],
        "difficulty": "medium",
        "category": "emergency",
        "subcategory": "debt_collection"
    }
]

def get_dataset_stats():
    """데이터셋 통계 정보 반환"""
    total_cases = len(dataset)
    
    # 난이도별 분류
    by_difficulty = {}
    for case in dataset:
        difficulty = case.get("difficulty", "unknown")
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
    
    # 카테고리별 분류
    by_category = {}
    for case in dataset:
        category = case.get("category", "unknown")
        by_category[category] = by_category.get(category, 0) + 1
    
    # 서브카테고리별 분류
    by_subcategory = {}
    for case in dataset:
        subcategory = case.get("subcategory", "unknown")
        by_subcategory[subcategory] = by_subcategory.get(subcategory, 0) + 1
    
    return {
        "total_cases": total_cases,
        "by_difficulty": by_difficulty,
        "by_category": by_category,
        "by_subcategory": by_subcategory
    }

def get_test_cases_by_category(category: str):
    """카테고리별 테스트 케이스 반환"""
    return [case for case in dataset if case.get("category") == category]

def get_test_cases_by_difficulty(difficulty: str):
    """난이도별 테스트 케이스 반환"""
    return [case for case in dataset if case.get("difficulty") == difficulty]

def get_test_cases_by_subcategory(subcategory: str):
    """서브카테고리별 테스트 케이스 반환"""
    return [case for case in dataset if case.get("subcategory") == subcategory]

if __name__ == "__main__":
    # 데이터셋 통계 출력
    stats = get_dataset_stats()
    print("🏦 KB금융 창구 행원 실무 RAG 테스트 데이터셋")
    print("=" * 60)
    print(f"📊 총 테스트 케이스: {stats['total_cases']}개")
    print(f"📈 난이도별: {stats['by_difficulty']}")
    print(f"🏷️ 카테고리별: {stats['by_category']}")
    print(f"🔍 서브카테고리별: {stats['by_subcategory']}")
    
    print("\n📋 카테고리별 상세 정보:")
    for category in stats['by_category']:
        cases = get_test_cases_by_category(category)
        print(f"\n  {category.upper()}:")
        for case in cases:
            print(f"    - {case['id']}: {case['query'][:50]}...")
