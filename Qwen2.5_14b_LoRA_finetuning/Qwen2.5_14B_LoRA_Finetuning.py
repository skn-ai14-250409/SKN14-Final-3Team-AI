# train.py

# export HF_TOKEN=hf_your_hf_token
# export HF_USERNAME=your_hf_username
# export WANDB_API_KEY=your_wandb_api_key (optional, for wandb logging)

# pip install -r requirements.txt

# [1단계] 환경 설정 및 라이브-러리 임포트
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from huggingface_hub import login
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

def main():
    # --- 1. 로그인 ---
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        login(token=hf_token)
        print("✅ Hugging Face 로그인 성공!")
    else:
        raise ValueError("Hugging Face 토큰을 찾을 수 없습니다. 'export HF_TOKEN=...'으로 환경 변수를 설정해주세요.")

    # [2단계] 모델 및 토크나이저 로드
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    print(f"--- 모델 '{model_id}' 로딩 완료 ---")

    # try: # RunPod에서 Liger 설치 후 주석 해제
    #     apply_liger_kernel_to_qwen2()
    #     print("--- Liger-Kernel 적용 완료 ---")
    # except ImportError:
    #     print("❗️ Liger-Kernel 적용 실패. 설치를 확인해주세요.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("--- 토크나이저 로딩 완료 ---")

    # [3단계] 데이터셋 로드
    hf_username = os.environ.get('HF_USERNAME')
    if not hf_username:
        raise ValueError("HF_USERNAME 환경 변수를 설정해주세요.")
    
    dataset_repo_id = "rucipheryn/combined-dataset-30K-final"
    final_dataset = load_dataset(dataset_repo_id)
    print(f"--- 데이터셋 '{dataset_repo_id}' 로드 완료 ---")
    print(final_dataset)

    # [4단계] LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_config)
    print("--- 모델에 LoRA 적용 완료 ---")
    lora_model.print_trainable_parameters()

    # [5단계] 훈련 계획 수립 (SFTConfig)
    
    report_to = "wandb" if os.environ.get('WANDB_API_KEY') else "none"
    if report_to == "wandb":
        os.environ['WANDB_PROJECT'] = "Qwen2.5_14B_LoRA_Finetuning"
        print("✅ wandb 연동이 활성화되었습니다.")

    # 🚀 최종 모델이 저장될 허깅페이스 저장소 이름
    final_model_repo_id = f"{hf_username}/Qwen2.5-14B-KB-Finance-LoRA-ep2"

    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        output_dir=final_model_repo_id, # 로컬 저장 경로와 허브 저장소 이름을 동일하게 설정
        report_to=report_to,
        run_name=f"{final_model_repo_id}-run",
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        push_to_hub=True, # 훈련이 끝나면 자동으로 허브에 업로드
    )

    # [6단계] 트레이너 생성
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
    )
    
    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["test"],
        data_collator=data_collator,
    )
    print("\n--- SFTTrainer 준비 완료 ---")

    # [7단계] 훈련 시작
    print("\n--- 파인튜닝을 시작합니다 ---")
    trainer.train()
    print("\n--- 파인튜닝 완료 ---")

    # [8단계] 모델 최종 저장 및 업로드
    # push_to_hub=True 옵션 덕분에 trainer.train()이 끝나면 자동으로 업로드된다.
    print(f"\n✅ 최종 모델이 Hugging Face Hub에 업로드되었습니다.")
    print(f"URL: https://huggingface.co/{final_model_repo_id}")


if __name__ == "__main__":
    main()