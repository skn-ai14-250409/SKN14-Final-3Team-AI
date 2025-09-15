# merge_model.py

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# --- 1. 설정 및 로그인 ---
load_dotenv() # .env 파일에서 HF_TOKEN, HF_USERNAME 로드
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME")
login(token=HF_TOKEN)

BASE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
LORA_ADAPTER_ID = f"{HF_USERNAME}/Qwen2.5-14B-KB-Finance-LoRA-v2"
MERGED_MODEL_REPO_ID = f"{HF_USERNAME}/Qwen2.5-14B-KB-Finance-LoRA-v2-Merged"

# --- 2. 모델 로드 및 병합 (RAM이 넉넉한 RunPod CPU/GPU 환경) ---
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto", # RunPod 환경에서는 auto로 설정
)

print("Loading and merging LoRA adapter...")
merged_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
merged_model = merged_model.merge_and_unload()
print("Merge complete!")

# --- 3. 병합된 모델과 토크나이저를 Hub에 업로드 ---
print("Uploading merged model to Hugging Face Hub...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
    )


# push_to_hub 사용
merged_model.push_to_hub(
    MERGED_MODEL_REPO_ID,
    private=True,
    token=HF_TOKEN,
    safe_serialization=True,      # safetensors로 저장
    max_shard_size="4GB",         # 파일 분할
)

tokenizer.push_to_hub(MERGED_MODEL_REPO_ID, private=True, token=HF_TOKEN)

print(f"\n✅ 병합된 모델이 성공적으로 업로드되었습니다!")
print(f"URL: https://huggingface.co/{MERGED_MODEL_REPO_ID}")