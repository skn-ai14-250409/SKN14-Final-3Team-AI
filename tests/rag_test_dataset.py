# tests/rag_test_dataset.py
"""
KB금융 RAG 시스템 테스트 데이터셋 (현재 시스템 맞춤)
- 4개 Intent 카테고리 기반
- 실제 PDF 문서 기반 질문들
- 현재 벡터 스토어에 있는 파일들 기준
"""

# 현재 시스템에 맞는 테스트 데이터셋
dataset = [
    # === 상품 관련 질문 (company_products) ===
    # 아래는 테스트 결과에서 확인된 실제 파일들 기반
    {
        "id": "product_001",
        "query": "KB 법인 예금담보 임직원대출의 대출한도는 얼마인가요?",
        "expected_answer": "법인이 담보로 제공한 정기예금의 95% 이내에서 대출이 가능합니다.",
        "expected_file": ["KB_법인_예금담보_임직원대출.pdf"],
        "difficulty": "easy",
        "category": "company_products",
        "subcategory": "개인 담보 전세대출"
    },
    {
        "id": "product_002",
        "query": "KB 기업당좌대출의 특징은 무엇인가요?",
        "expected_answer": "기업의 운영자금 조달을 위한 단기 대출상품으로, 당좌거래약정 체결 기업이 대상입니다.",
        "expected_file": ["KB_기업당좌대출.pdf"],
        "difficulty": "medium",
        "category": "company_products",
        "subcategory": "기업 대출"
    },
    {
        "id": "product_003",
        "query": "KB 햇살론 119의 대출 조건은 어떻게 되나요?",
        "expected_answer": "저소득층을 위한 정부지원 대출로, 연소득과 신용점수 기준이 있습니다.",
        "expected_file": ["KB_햇살론_119.pdf"],
        "difficulty": "easy",
        "category": "company_products",
        "subcategory": "기업 대출"
    },
    {
        "id": "product_004",
        "query": "KB 동반성장협약 상생대출의 대출신청자격은 어떻게 되나요?",
        "expected_answer": "당행과 협약을 체결한 대기업이 발급한 「대출추천서」를 제출하는 협력기업이어야 합니다.",
        "expected_file": ["KB_동반성장협약_상생대출.pdf"],
        "difficulty": "medium",
        "category": "company_products",
        "subcategory": "기업 대출"
    },
    
    # === 내규 관련 질문 (company_rules) ===
    {
        "id": "rules_001",
        "query": "여신업무 처리 시 준수해야 할 기본 원칙은 무엇인가요?",
        "expected_answer": "여신업무는 공정성, 투명성, 안정성을 기본 원칙으로 하며, 고객의 상환능력을 정확히 평가하여 적정한 대출을 실행해야 합니다.",
        "expected_file": ["여신기본강령_및_모범규준.pdf"],
        "difficulty": "easy",
        "category": "company_rules",
        "subcategory": "국민은행"
    },
    {
        "id": "rules_002",
        "query": "임직원이 이해관계자로부터 금품을 받으면 어떻게 되나요?",
        "expected_answer": "임직원이 이해관계자로부터 금품을 받는 것은 엄격히 금지되며, 위반 시 징계처분을 받게 됩니다.",
        "expected_file": ["KB금융_반부패_뇌물정책.pdf"],
        "difficulty": "medium",
        "category": "company_rules",
        "subcategory": "국민은행"
    },
    {
        "id": "rules_003",
        "query": "KB금융그룹의 윤리강령에서 정하는 기본 가치는 무엇인가요?",
        "expected_answer": "정직, 신뢰, 고객중심, 상호존중, 사회적 책임을 기본 가치로 합니다.",
        "expected_file": ["윤리강령.pdf"],
        "difficulty": "easy",
        "category": "company_rules",
        "subcategory": "국민은행"
    },
    
    # === 법률/규정 관련 질문 (industry_policies_and_regulations) ===
    {
        "id": "law_001",
        "query": "은행법에서 정하는 여신한도 규제는 어떻게 되나요?",
        "expected_answer": "은행법에 따라 동일인에 대한 여신한도는 은행 자기자본의 일정 비율을 초과할 수 없으며, 대기업집단에 대해서는 별도의 한도 규제가 적용됩니다.",
        "expected_file": ["은행법.pdf"],
        "difficulty": "hard",
        "category": "industry_policies_and_regulations",
        "subcategory": "공통"
    },
    {
        "id": "law_002",
        "query": "금융소비자보호법의 주요 내용은 무엇인가요?",
        "expected_answer": "금융소비자의 권익 보호를 위해 금융상품 판매 시 적합성·적정성 원칙 준수, 설명의무 이행, 불공정영업행위 금지 등을 규정하고 있습니다.",
        "expected_file": ["금융소비자보호법.pdf"],
        "difficulty": "medium",
        "category": "industry_policies_and_regulations",
        "subcategory": "공통"
    },
    
    # === 일반 금융 FAQ (general_banking_FAQs) ===
    {
        "id": "faq_001",
        "query": "금리가 무엇인가요?",
        "expected_answer": "금리는 돈을 빌리거나 예치할 때 적용되는 이자율을 의미합니다. 연간 몇 퍼센트의 이자를 지불하거나 받을지를 나타내는 지표입니다.",
        "expected_file": [],
        "difficulty": "easy",
        "category": "general_banking_FAQs",
        "subcategory": "기본개념"
    },
    {
        "id": "faq_002",
        "query": "담보대출과 신용대출의 차이점은 무엇인가요?",
        "expected_answer": "담보대출은 부동산 등의 담보를 제공하여 받는 대출로 금리가 낮고 한도가 높습니다. 신용대출은 담보 없이 개인의 신용도만으로 받는 대출로 금리가 상대적으로 높고 한도가 제한적입니다.",
        "expected_file": [],
        "difficulty": "easy",
        "category": "general_banking_FAQs",
        "subcategory": "기본개념"
    },
    {
        "id": "faq_003",
        "query": "예금자보호제도란 무엇인가요?",
        "expected_answer": "예금자보호제도는 은행이 파산하거나 영업이 정지될 경우 예금자의 예금을 보호하는 제도로, 1인당 최대 5천만원까지 보호됩니다.",
        "expected_file": [],
        "difficulty": "medium",
        "category": "general_banking_FAQs",
        "subcategory": "기본개념"
    }
]

def get_dataset_stats():
    """데이터셋 통계 정보 반환"""
    stats = {
        "total_questions": len(dataset),
        "categories": {},
        "difficulties": {},
        "subcategories": {}
    }
    
    for item in dataset:
        # 카테고리별 통계
        category = item.get("category", "unknown")
        if category not in stats["categories"]:
            stats["categories"][category] = 0
        stats["categories"][category] += 1
        
        # 난이도별 통계
        difficulty = item.get("difficulty", "unknown")
        if difficulty not in stats["difficulties"]:
            stats["difficulties"][difficulty] = 0
        stats["difficulties"][difficulty] += 1
        
        # 서브카테고리별 통계
        subcategory = item.get("subcategory", "unknown")
        if subcategory not in stats["subcategories"]:
            stats["subcategories"][subcategory] = 0
        stats["subcategories"][subcategory] += 1
    
    return stats

def get_test_cases_by_category(category: str):
    """카테고리별 테스트 케이스 반환"""
    return [item for item in dataset if item.get("category") == category]

def get_test_cases_by_difficulty(difficulty: str):
    """난이도별 테스트 케이스 반환"""
    return [item for item in dataset if item.get("difficulty") == difficulty]

def get_test_cases_by_subcategory(subcategory: str):
    """서브카테고리별 테스트 케이스 반환"""
    return [item for item in dataset if item.get("subcategory") == subcategory]