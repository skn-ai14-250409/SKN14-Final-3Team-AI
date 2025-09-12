# 환경변수 설정 예시
# export HF_TOKEN=               # Hugging Face 인증 토큰(선택: CLI 로그인 시 생략 가능)
# export WANDB_API_KEY=       # wandb 로깅용 API 키(선택)
# export HF_USERNAME=rucipheryn                                       # Hugging Face 사용자명(선택: CLI 로그인 시 생략 가능)

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def setup_environment():
    """Hugging Face 및 wandb 환경을 설정하고 (hf_username, report_to)를 반환합니다."""
    hf_token = os.environ.get("HF_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    # 토큰이 있으면 로그인, 없으면 기존 CLI 로그인/캐시를 사용
    if hf_token:
        login(token=hf_token)
        print("Hugging Face 로그인 성공")
    else:
        print("HF_TOKEN이 설정되지 않아, 기존 CLI 로그인/캐시를 사용합니다")

    # 요청에 따라 항상 이 네임스페이스로 푸시
    hf_username = "rucipheryn"

    # wandb 설정
    report_to = "none"
    if wandb_api_key:
        report_to = "wandb"
        os.environ["WANDB_PROJECT"] = "sk-networks-ai-camp-final"
        print("wandb 로깅 활성화")
    else:
        print("WANDB_API_KEY 미설정: wandb 로깅 비활성화(report_to=none)")

    return hf_username, report_to


def configure_h100_optimizations():
    """H100에서 권장되는 수학 설정(TF32/고정밀 matmul)을 활성화합니다."""
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print("H100/TF32 최적화 활성화")
        else:
            print("CUDA 미사용: H100 최적화 스킵")
    except Exception as e:
        print(f"H100 최적화 설정 중 오류: {e}")


def maybe_apply_liger():
    """환경변수 USE_LIGER(1/true/yes/y)일 때 Liger-Kernel을 선택적으로 적용합니다."""
    flag = os.environ.get("USE_LIGER", "0").strip().lower()
    enabled = flag in ("1", "true", "yes", "y")
    if not enabled:
        print("USE_LIGER 비활성: Liger-Kernel 적용 스킵")
        return
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2
        apply_liger_kernel_to_qwen2()
        print("Liger-Kernel 적용 완료")
    except Exception as e:
        print(f"Liger-Kernel 적용 실패: {e}")


def load_model_and_tokenizer(model_id: str):
    """권장 기본값으로 모델과 토크나이저를 로드합니다."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        use_cache=False,  # gradient_checkpointing=True와 함께 사용하려면 False로 설정
    )
    print(f"--- 모델 '{model_id}' 로드 완료 ---")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("--- 토크나이저 준비 완료 ---")
    return model, tokenizer


def build_lora_config() -> LoraConfig:
    """Causal LM을 위한 LoRA 설정을 생성합니다."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

def main():
    # 기본 환경 설정
    hf_username, report_to = setup_environment()
    configure_h100_optimizations()
    MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
    DATASET_ID = f"{hf_username}/combined-dataset-30K-final-v2"
    OUTPUT_DIR = "Qwen2.5-14B-KB-Finance-LoRA-v2"
    FINAL_HUB_REPO_ID = f"{hf_username}/{OUTPUT_DIR}"

    # 모델/토크나이저/데이터셋 로드
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    maybe_apply_liger()
    dataset = load_dataset(DATASET_ID)

    # --- 💡 핵심 수정 시작 💡 ---
    # 1. 모델 준비: PEFT 라이브러리의 헬퍼 함수를 사용해 모델을 사전 처리한다.
    #    (양자화를 안 해도 이 함수를 쓰는 것이 gradient_checkpointing 안정성에 도움이 됨)
    model = prepare_model_for_kbit_training(model)
    
    # 2. LoRA 적용
    lora_model = get_peft_model(model, build_lora_config())
    # ⛔️ lora_model.train()은 여기서 호출하지 않는다. Trainer가 알아서 처리하도록 둔다.
    # --- 💡 핵심 수정 끝 💡 ---

    print("\n--- LoRA 어댑터 적용 완료 ---")
    lora_model.print_trainable_parameters()

    # 학습 설정 구성
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        hub_model_id=FINAL_HUB_REPO_ID,
        report_to=report_to,
        run_name=f"{OUTPUT_DIR}-run",
        num_train_epochs=2,
        learning_rate=2e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=300,
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=2048,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=True,
        hub_private_repo=False,
    )

    # 트레이너 구성
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
    )
    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        data_collator=data_collator,
    )
    print("\n--- SFTTrainer 준비 완료 ---")

    # 학습 시작
    print("\n--- 파인튜닝 시작 ---")
    trainer.train()
    print("\n--- 파인튜닝 완료 ---")
    print(f"\n모델이 Hugging Face Hub로 푸시되었습니다: {FINAL_HUB_REPO_ID}")


if __name__ == "__main__":
    main()