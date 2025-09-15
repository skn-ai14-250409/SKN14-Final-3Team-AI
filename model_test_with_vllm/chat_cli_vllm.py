# chat_cli_vllm.py

import os
from openai import OpenAI # 💡 openai 라이브러리를 사용

# --- 1. vLLM 서버 정보 설정 ---
# vLLM은 OpenAI API와 호환되는 서버를 생성한다.
# 따라서 OpenAI 클라이언트를 그대로 사용할 수 있다.
client = OpenAI(
    base_url="http://localhost:8000/v1", # 💡 접속 주소를 우리 vLLM 서버로 변경
    api_key="NOT_USED"                   # 💡 로컬 서버라 API 키는 필요 없음
)

# vLLM 서버에 올라가 있는 모델의 ID
MODEL_ID = "rucipheryn/Qwen2.5-14B-KB-Finance-LoRA-v2-Merged"


# --- 2. 메인 대화 루프 ---
def main():
    print("\n=============================================")
    print("  KB 금융 챗봇 터미널 (vLLM 구동)")
    print("  (종료하려면 'exit' 또는 'quit'를 입력하세요)")
    print("=============================================")

    # 대화 기록을 저장할 리스트
    messages = [
    {
        "role": "system",
        "content": (
            "You are a professional financial assistant for KB Financial Group. "
            "CRITICAL: Write ALL answers in Korean ONLY, using a friendly informal tone (banmal) appropriate for a colleague. "
            "Do NOT use English in the output unless the user explicitly asks; keep the entire response in Korean.\n\n"

            "## Style & Structure\n"
            "- Use Markdown with clear sections and bullets.\n"
            "- Always follow this layout:\n"
            "  1) ## 요약 — one-line takeaway\n"
            "  2) ## 핵심 포인트 — 3–5 bullets\n"
            "  3) ## 상세 설명 — concepts → context → steps/examples\n"
            "  4) ## 표 (필요 시) — simple table\n"
            "  5) ## 추가로 필요한 정보 (선택) — 1–3 clarifying items\n\n"

            "## Accuracy & Safety\n"
            "- Prioritize factual accuracy; if unknown or uncertain, say '모르겠다' and state what info is needed.\n"
            "- When discussing Korean retail banking/loans, reflect the local context (e.g., Bank of Korea base rate as a policy rate, COFIX as a common floating index) when relevant, and include dates like YYYY-MM-DD for time-sensitive figures.\n"
            "- Avoid inventing numbers or sources; do not include sensitive personal data.\n\n"

            "## Formatting Rules (in Korean output)\n"
            "- Currency: 원(₩) with thousand separators (e.g., 1,234,000원).\n"
            "- Rates: 2.5% ; changes as percentage points (e.g., +0.25%p).\n"
            "- Define financial acronyms on first use (예: APR, APY, COFIX, 기준금리).\n"
        )
    }
]

    while True:
        # 1. 사용자로부터 질문 입력받기
        user_input = input("\n👤 당신: ")
        if user_input.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
            
        # 2. 대화 기록에 사용자 메시지 추가
        messages.append({"role": "user", "content": user_input})

        print("\n🤖 KB 챗봇이 답변을 생성 중입니다...")
        
        try:
            # 3. vLLM 서버에 API 요청
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            
            # 4. 답변 추출 및 출력
            assistant_response = response.choices[0].message.content
            print(f"\n🤖 KB 챗봇: {assistant_response}")
            
            # 5. 대화 기록에 모델 답변 추가 (연속적인 대화를 위해)
            messages.append({"role": "assistant", "content": assistant_response})

        except Exception as e:
            print(f"\n❗️ 서버와 통신 중 오류가 발생했습니다: {e}")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    main()