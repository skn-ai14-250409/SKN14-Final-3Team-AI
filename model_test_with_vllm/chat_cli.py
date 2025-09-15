# chat_cli.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
from dotenv import load_dotenv

# --- 1. 환경 설정 및 .env 파일 로드 ---
# load_dotenv() 함수가 .env 파일을 찾아서 환경 변수로 자동으로 로드해준다.
load_dotenv()

# 💡 .env 또는 export로 설정된 환경 변수를 불러온다.
#    get() 메소드의 두 번째 인자(기본값)는 이제 삭제해도 안전하다.
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")
assert HF_USERNAME, "환경변수 HF_USERNAME이 비어 있습니다 (.env 확인)."
MODEL_ID = f"{HF_USERNAME}/Qwen2.5-14B-KB-Finance-LoRA-v2-Merged"

# --- 2. 모델 및 토크나이저 로드 ---
def load_model():
    """
    파인튜닝되고 병합된 최종 모델과 토크나이저를 로드하는 함수.
    RunPod의 GPU 환경을 가정한다.
    """
    print(f"'{MODEL_ID}' 모델을 로드합니다... (시간이 소요될 수 있습니다)")

    # 양자화 없이, bfloat16으로 모델을 로드
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("✅ 모델과 토크나이저 로드 완료!")
    return model, tokenizer

# --- 3. 메인 대화 루프 ---
def main():
    model, tokenizer = load_model()

    print("\n=============================================")
    print("  KB 금융 챗봇 터미널에 오신 것을 환영합니다!")
    print("  (종료하려면 'exit' 또는 'quit'를 입력하세요)")
    print("=============================================")

    while True:
        # 1. 사용자로부터 질문 입력받기
        user_input = input("\n👤 당신: ")
        if user_input.lower() in ["exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        # 2. Qwen2 템플릿에 맞춰 프롬프트 생성
        messages = [
            {"role": "system", "content": "KB 금융 도메인 보조역. 추측 금지, 근거 우선. 반말로 간결하게."},
            {"role": "user", "content": user_input},
            ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. 모델 입력 준비
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        # 4. 답변 생성
        # GenerationConfig를 사용해 생성 파라미터를 명시적으로 관리
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        print("\n🤖 KB 챗봇이 답변을 생성 중입니다...")
        outputs = model.generate(**inputs, generation_config=generation_config)

        # 5. 결과 디코딩 및 출력
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 프롬프트를 제외한 순수 답변만 추출
        try:
            assistant_response = response_text.split("<|im_start|>assistant", 1)[-1]
            assistant_response = assistant_response.split("<|im_end|>", 1)[0].strip()
            print(f"\n🤖 KB 챗봇: {assistant_response}")
        except:
            print(f"\n❗️ 답변 추출에 실패했습니다. 전체 출력:\n{response_text}")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # Hugging Face 토큰이 있는지 다시 한번 확인
    if not HF_TOKEN:
        print("경고: .env 파일이나 환경 변수에 HF_TOKEN이 설정되지 않았습니다.")
        # 필요하다면 여기서 프로그램을 종료할 수도 있다.
        # exit()
    main()