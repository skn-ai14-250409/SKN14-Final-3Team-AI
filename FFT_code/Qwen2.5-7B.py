# pip install unsloth
# pip install transformers==4.55.4
# pip install wandb
# export HF_TOKEN="자신의 API 넣기!"
# export WANDB_API_KEY="자신의 API 넣기!"
# train_full_model.py

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from transformers import AutoTokenizer

# ===================================================================================
# 1. 설정 (CONFIGURATION)
# ===================================================================================
# Hugging Face Hub 사용자 이름 또는 조직 이름
HF_USERNAME = "sssssungjae" # <--- 🙋‍♂️ 여기에 본인의 Hugging Face ID를 입력하세요.

# 🌟 WandB 설정
WANDB_PROJECT_NAME = "qwen2.5-7b-finance-full-finetune" # <--- 🙋‍♂️ WandB에 표시될 프로젝트 이름

# 모델 및 토크나이저 설정
BASE_MODEL_NAME = "unsloth/Qwen2.5-7B" # <--- 🌟 모델 변경
MAX_SEQ_LENGTH = 1024

# 🚫 LoRA 설정은 Full-Finetuning 시 필요 없으므로 제거합니다.

# 데이터셋 설정
DATASET_REPO_NAME = "sssssungjae/finance-kb-mixed-dataset-final"
DATASET_SPLIT = "train"
DATASET_TEXT_FIELD = "text"  # 실제 컬럼명과 다르면 수정하세요.

# 평가 데이터셋 설정 (옵션)
# 환경변수로 덮어쓰기 가능: EVAL_DATASET_REPO, EVAL_DATASET_SPLIT
EVAL_DATASET_REPO_NAME = os.environ.get("EVAL_DATASET_REPO", "")
EVAL_DATASET_SPLIT = os.environ.get("EVAL_DATASET_SPLIT", "validation")

# 학습 하이퍼파라미터
TRAINING_EPOCHS = 1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 2e-5
# H100에서는 8-bit 옵티마이저보다 fused AdamW가 일반적으로 더 빠릅니다.
OPTIMIZER = "adamw_torch_fused"

# 저장 및 업로드 설정
LOCAL_FULL_MODEL_PATH = "full_model"
LOCAL_GGUF_PATH = "model_gguf"
HUB_FULL_REPO = f"{HF_USERNAME}/qwen2_5-7b-finance-full"
HUB_GGUF_REPO = f"{HF_USERNAME}/qwen2_5-7b-finance-full-gguf"

# ===================================================================================
# 2. 데이터 준비 함수 (필요 시 확장 가능)
# ===================================================================================


# ===================================================================================
# 3. 모델 학습 함수
# ===================================================================================
def train_model(train_dataset, tokenizer, eval_dataset=None):
    """Full-Finetuning으로 모델을 학습합니다. (H100 최적화)"""
    print("\n step 2: Full-Finetuning 모델 로딩 및 학습 시작...")

    # 🌟 Full-Finetuning으로 모델 및 토크나이저 로드
    model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        full_finetuning=True,  # <--- 🌟 Full-Finetuning 활성화
        torch_dtype=torch.bfloat16,
    )

    # 학습 최적화 설정 (H100)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # special tokens 추가 시 임베딩 크기 재조정 및 캐시 비활성화
    if hasattr(tokenizer, "vocab") or hasattr(tokenizer, "get_vocab"):
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 🚫 LoRA 어댑터를 추가하는 get_peft_model 함수는 사용하지 않습니다.

    # SFTTrainer 설정
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 옵션: 평가 데이터셋
        # 사전 토크나이즈된 데이터셋을 사용하므로 텍스트 필드 지정 불필요
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=100,
            num_train_epochs=TRAINING_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            optim=OPTIMIZER,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
            bf16=True,
            fp16=False,
            tf32=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            # 평가/체크포인트 설정
            eval_strategy=("steps" if eval_dataset is not None else "no"),
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=(eval_dataset is not None),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )

    # 🌟 WandB 프로젝트 이름 환경 변수 설정
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

    # 학습 실행
    trainer.train()
    
    print("✅ 모델 학습 완료.")
    return model, tokenizer

# ===================================================================================
# 4. 모델 저장 및 업로드 함수
# ===================================================================================
def save_and_upload_models(model, tokenizer):
    """학습된 전체 모델을 저장하고 Hub에 업로드합니다."""
    print("\n step 3: 모델 저장 및 업로드 시작...")

    try:
        login(token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        print("Hugging Face 로그인이 필요합니다. export HF_TOKEN='내토큰' 명령어로 토큰을 설정해주세요.")
        print(f"로그인 오류: {e}")
        return

    # 1. Full-Finetuning된 모델 저장 및 업로드 (16비트)
    print("\n(1/2) Full-Finetuning 모델 저장 및 업로드 중...")
    model.save_pretrained(LOCAL_FULL_MODEL_PATH) # 로컬에 저장
    tokenizer.save_pretrained(LOCAL_FULL_MODEL_PATH)
    model.push_to_hub(HUB_FULL_REPO, token=True) # Hub에 업로드
    tokenizer.push_to_hub(HUB_FULL_REPO, token=True)
    print(f"✅ 전체 모델 업로드 완료: {HUB_FULL_REPO}")

    # 2. GGUF 모델 저장 및 업로드 (모델/버전에 따라 미지원일 수 있음)
    try:
        print("\n(2/2) GGUF 모델 저장 및 업로드 중...")
        model.save_pretrained_gguf(
            LOCAL_GGUF_PATH,
            tokenizer,
            quantization_method=["q4_k_m"],
        )
        model.push_to_hub_gguf(
            HUB_GGUF_REPO,
            tokenizer,
            quantization_method=["q4_k_m"],
            token=True,
        )
        print(f"✅ GGUF 모델 업로드 완료: {HUB_GGUF_REPO}")
    except Exception as e:
        print(f"⚠️ GGUF 내보내기/업로드가 지원되지 않거나 실패했습니다: {e}")

# ===================================================================================
# 5. 메인 실행 블록
# ===================================================================================
if __name__ == "__main__":
    # WandB 프로젝트 이름 먼저 설정
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

    # 토크나이저: Instruct 템플릿을 Base 토크나이저에 이식
    print("  step 0: 베이스 모델 토크나이저에 채팅 템플릿 설정 중...")
    instruct_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-7B-Instruct")
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    base_tokenizer.chat_template = instruct_tokenizer.chat_template
    base_tokenizer.add_special_tokens(
        instruct_tokenizer.special_tokens_map,
        replace_additional_special_tokens=True,
    )
    print("✅ 채팅 템플릿 설정 완료.")
    
    # 1. 데이터 준비 (train + optional eval)
    print("\n  step 1: 허깅페이스 Hub에서 전처리된 데이터셋 로딩...")
    final_dataset = load_dataset(DATASET_REPO_NAME, split=DATASET_SPLIT)
    if DATASET_TEXT_FIELD not in final_dataset.column_names:
        raise ValueError(
            f"데이터셋에 '{DATASET_TEXT_FIELD}' 컬럼이 없습니다. 현재 컬럼: {final_dataset.column_names}"
        )
    print(f"✅ 데이터셋 로딩 완료. 크기: {len(final_dataset)}")

    # 평가 데이터셋 로딩 (있으면 사용, 없으면 동일 저장소의 eval/validation 시도)
    eval_dataset = None
    if EVAL_DATASET_REPO_NAME:
        try:
            print("  step 1b: 평가 데이터셋 로딩...")
            eval_dataset = load_dataset(EVAL_DATASET_REPO_NAME, split=EVAL_DATASET_SPLIT)
            if DATASET_TEXT_FIELD not in eval_dataset.column_names:
                raise ValueError(
                    f"평가 데이터셋에 '{DATASET_TEXT_FIELD}' 컬럼이 없습니다. 현재 컬럼: {eval_dataset.column_names}"
                )
            print(f"✅ 평가 데이터셋 로딩 완료. 크기: {len(eval_dataset)}")
        except Exception as e:
            print(f"⚠️ 평가 데이터셋 로딩 실패, 평가를 건너뜁니다: {e}")
    else:
        # 환경변수가 없다면 동일 저장소에서 공통 eval 스플릿명 순차 시도
        for cand_split in ("validation", "eval", "valid"):
            try:
                print(f"  step 1b: 동일 저장소에서 평가 스플릿('{cand_split}') 로딩 시도...")
                cand_eval = load_dataset(DATASET_REPO_NAME, split=cand_split)
                if DATASET_TEXT_FIELD not in cand_eval.column_names:
                    raise ValueError(
                        f"평가 데이터셋에 '{DATASET_TEXT_FIELD}' 컬럼이 없습니다. 현재 컬럼: {cand_eval.column_names}"
                    )
                eval_dataset = cand_eval
                print(f"✅ 평가 데이터셋 로딩 완료. 스플릿: '{cand_split}', 크기: {len(eval_dataset)}")
                break
            except Exception as e:
                print(f"  ↪️ '{cand_split}' 스플릿 로딩 실패: {e}")
        if eval_dataset is None:
            print(" 평가 스플릿을 찾지 못해 평가를 생략합니다. (환경변수로 지정 가능)")
    
    # 1c. 사전 토크나이즈: input_ids / attention_mask 생성
    print("  step 1c: 데이터셋 토크나이즈 중 (input_ids/attention_mask 생성)...")
    def _tokenize_batch(batch):
        texts = batch[DATASET_TEXT_FIELD]
        if isinstance(texts, str):
            texts = [texts]
        enc = base_tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    final_dataset = final_dataset.map(_tokenize_batch, batched=True, desc="Tokenizing train")
    if eval_dataset is not None:
        try:
            eval_dataset = eval_dataset.map(_tokenize_batch, batched=True, desc="Tokenizing eval")
        except Exception as e:
            print(f"⚠️ 평가 데이터셋 토크나이즈 중 오류: {e}. 평가를 건너뜁니다.")
            eval_dataset = None
    
    # 2. 모델 학습 (이후 코드는 변경 없음)
    trained_model, trained_tokenizer = train_model(
        train_dataset=final_dataset, tokenizer=base_tokenizer, eval_dataset=eval_dataset
    )
    
    # 3. 모델 저장 및 업로드
    save_and_upload_models(model=trained_model, tokenizer=trained_tokenizer)
    
    print("\n🎉 모든 파인튜닝 과정이 완료되었습니다!")
