# -*- coding: utf-8 -*-
###############################################################################
# Qwen3-8B + Unsloth QLoRA 파인튜닝 스크립트 (주석 정리판)
# - 동작/로직은 그대로 유지하고, 주석과 설명만 정리했습니다.
# - 전체 흐름: 4bit 모델 로드 → LoRA 부착 → 데이터 준비/혼합 → (옵션) 데이터셋 업로드
#             → SFTTrainer 학습 → 로컬/Hub 저장 → 추론 → 병합 모델/GGUF 업로드
# - 실제 환경에 맞는 패키지 설치 및 토큰 설정(HF_TOKEN 등)을 사전에 준비하세요.
###############################################################################

# Commented out IPython magic to ensure Python compatibility.
# 해당 라이브러리를 설치해야합니다!
# pip install unsloth transformers==4.55.4

"""### Unsloth (모델 로드/준비)"""

from unsloth import FastLanguageModel
import torch

# -----------------------------------------------------------------------------
# 1) 모델 후보(정보용) 및 베이스 모델 로드
# -----------------------------------------------------------------------------

# 베이스 모델(4bit) 로드: QLoRA 기반 학습을 위한 설정
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",
    max_seq_length = 512,   # 컨텍스트 길이 (길수록 메모리 사용 증가)
    load_in_4bit = True,    # 4bit 양자화로 메모리 절감
    load_in_8bit = False,   # 8bit는 정확도↑/메모리↑
    full_finetuning = False,# QLoRA 시 False 유지
    # token = "hf_...",     # 게이트드 모델이면 토큰 필요
)

"""
LoRA 어댑터 부착
- 전체 파라미터의 1~10%만 업데이트 → 효율적인 미세조정
- use_gradient_checkpointing = "unsloth" 로 긴 컨텍스트 대응
"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

"""<a name="Data"></a>
### Data
저희는 Qwen3의 reasoning 기능을 사용하지 않을 것이기 때문에 데이터 또한 non reasoning 데이터 2가지로 구성하였음.

1. 저희는 금융지식을 학습하기 위해 [aiqwe/FinShibainu](https://huggingface.co/datasets/aiqwe/FinShibainu/viewer/qa) 데이터셋을 사용하였고 이것은 [KRX](https://krxbench.koscom.co.kr/) (KRX LLM 경진대회 리더보드에서 우수상을 수상한 shibainu24의 학습데이터)여기서 활용된 데이터를 가져왔습니다.
간단하게 QA로 이뤄져있으며 answer_B 칼럼을 답으로 사용함.

2. 저희는 또한 [davidkim205/kollm-converations](https://huggingface.co/datasets/davidkim205/kollm-converations/viewer/default/train?row=0&views%5B%5D=train) ShareGPT style의 한국어 대화 데이터셋을 사용하였으며 unsloth의 예시 코드처럼 형식을 맞추고 한국어를 학습 하기위해 사용하였습니다.
기존 코드의 데이터셋은 multiturn 데이터셋이기 때문에 멀티턴을 학습하기 위해선 데이터셋을 멀티셋으로 구성하는것도 고려해봐야함.
기존데이터셋:[mlabonne/FineTome-100k](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdatasets%2Fmlabonne%2FFineTome-100k)
"""

from datasets import load_dataset
non_reasoning_Fin_dataset = load_dataset("aiqwe/FinShibainu", 'qa')
non_reasoning_dataset = load_dataset("davidkim205/kollm-converations", split = "train")


"""
대화 형식 변환(1): ShareGPT 스타일 데이터셋 정규화 후 템플릿 적용
- Unsloth `standardize_sharegpt`로 대화 형식 표준화
"""

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(non_reasoning_dataset)

non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)


"""
대화 형식 변환(2): 금융 QA → 대화 포맷(conversations)으로 변환 후 템플릿 적용
- question → user, answer_B → assistant 매핑
- .map()으로 전체 적용 → 템플릿으로 최종 문자열 생성
"""

def generate_conversation(examples):
    # FinShibainu QA → conversations 구조로 변환
    questions  = examples["question"]
    answers = examples["answer_B"]
    conversations = []
    for question, answer in zip(questions, answers):
        conversations.append([
            {"role" : "user",      "content" : question},
            {"role" : "assistant", "content" : answer},
        ])
    return { "conversations": conversations, }

# 먼저 'train' 분할을 명시적으로 선택합니다.
mapped_dataset = non_reasoning_Fin_dataset['train'].map(
    generate_conversation,
    batched = True
)

# 그 결과(mapped_dataset)에서 'conversations' 컬럼을 가져옵니다.
non_reasoning_Fin_conversations = tokenizer.apply_chat_template(
    mapped_dataset["conversations"],
    tokenize = False,
)

# (선택 사항) 결과 확인
print(non_reasoning_Fin_conversations[0])

"""데이터셋 길이 확인"""

print(len(non_reasoning_Fin_conversations))
print(len(non_reasoning_conversations))

"""
대화 데이터가 더 길기 때문에 혼합 비율을 조절하여
금융 특화 성능과 대화 능력의 균형을 맞춥니다.
여기서는 금융 QA 75% : 대화 25%로 설정.
"""

chat_percentage = 0.25

"""대화 데이터셋에서 chat_percentage 비율에 맞춰 샘플링"""

import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(non_reasoning_Fin_conversations)*(chat_percentage/(1 - chat_percentage))),
    random_state = 2407,
)
print(len(non_reasoning_Fin_conversations))
print(len(non_reasoning_subset))
print(len(non_reasoning_subset) / (len(non_reasoning_subset) + len(non_reasoning_Fin_conversations)))

"""두 데이터셋 결합 후 셔플 및 Dataset 변환"""

data = pd.concat([
    pd.Series(non_reasoning_Fin_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

data

len(combined_dataset)

"""## 학습 시간/자원에 따라 학습 샘플 수 제한"""

# 3단계: 최종 학습량 조절 (이 부분을 추가)
# 이미 잘 섞여 있으므로, 앞에서부터 2만 건을 순서대로 잘라내도
# 무작위 샘플링과 동일한 효과를 가집니다.
final_training_dataset = combined_dataset.select(range(20000))

print(f"원본 데이터셋 크기: {len(combined_dataset)}")
print(f"조정 후 최종 학습 데이터셋 크기: {len(final_training_dataset)}")

"""## 허깅페이스에 데이터셋 올리기(옵션)
- 전체/학습 서브셋을 각각 업로드. 환경에 HF 토큰 필요.
"""

from datasets import Dataset
import pandas as pd # Import pandas

# Convert the pandas Series to a Hugging Face Dataset
# Convert Series to DataFrame before passing to Dataset.from_pandas
hf_dataset = Dataset.from_pandas(pd.DataFrame(data))

# Replace "YOUR_HF_USERNAME/your-dataset-name" with your desired repo name and dataset name on Hugging Face
dataset_repo_name = "sssssungjae/finance-conversations-dataset"

print(f"Pushing full dataset to {dataset_repo_name}...")
hf_dataset.push_to_hub(dataset_repo_name, token = True)
print("✅ Full dataset uploaded.")

# Convert the final training dataset to a Hugging Face Dataset
# final_training_dataset is already a Dataset, no need to convert again.
# If it were a Series, we would do pd.DataFrame(final_training_dataset)
hf_final_training_dataset = final_training_dataset # It's already a Dataset

# Define a separate repo name for the final training dataset
final_training_dataset_repo_name = "sssssungjae/finance-conversations-training-Mix"

print(f"Pushing final training dataset to {final_training_dataset_repo_name}...")
hf_final_training_dataset.push_to_hub(final_training_dataset_repo_name, token = True)
print("✅ Final training dataset uploaded.")

"""
<a name="Train"></a>
### SFT 학습 설정 및 실행
- TRL SFTTrainer로 학습. 주요 하이퍼파라미터는 아래 args에서 제어합니다.
"""

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = final_training_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 100,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 1000,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

"""# 로컬 저장(LoRA + 토크나이저)"""

# LoRA 어댑터와 토크나이저 설정을 "lora_model"이라는 폴더에 저장합니다.
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

"""# 허깅페이스 업로드"""

from huggingface_hub import login

login()

# 내 허깅페이스 ID와 원하는 모델 이름으로 변경하세요.
hf_repo_name = "qwen3-8B-fin-adapter"

# 모델 어댑터 업로드
model.push_to_hub(hf_repo_name, token = True)

# 토크나이저 업로드
tokenizer.push_to_hub(hf_repo_name, token = True)


"""### VLLM용 float16 모델 저장하기

`float16`으로 바로 저장하는 것을 지원하는 코드. float16을 원하시면 `merged_16bit`를, `int4`를 원하면 `merged_4bit`를
"""
# 16비트 병합 모델 업로드 (고성능 배포용)
# Replace "YOUR_HF_USERNAME/qwen3-8b-finance-16bit" with your desired repo name
repo_name_16bit = "sssssungjae/qwen3-8b-finance-16bit"

print(f"Pushing 16-bit merged model to {repo_name_16bit}...")
model.push_to_hub_merged(repo_name_16bit, tokenizer, save_method = "merged_16bit", token = True)
print("✅ Done.")

# 4비트 병합 모델 업로드 (자원 효율적 배포용)
# Replace "YOUR_HF_USERNAME/qwen3-8b-finance-4bit" with your desired repo name
repo_name_4bit = "sssssungjae/qwen3-8b-finance-4bit"

print(f"Pushing 4-bit merged model to {repo_name_4bit}...")
model.push_to_hub_merged(repo_name_4bit, tokenizer, save_method = "merged_4bit_forced", token = True)
print("✅ Done.")

"""### GGUF / llama.cpp 변환하기
이제 GGUF / llama.cpp 포맷을 정식으로 지원합니다! 저희는 `llama.cpp`를 복제하며, 기본적으로 `q8_0` 방식으로 저장합니다. `q4_k_m`과 같은 모든 방식이 허용됩니다. 로컬에 저장하려면 save_pretrained_gguf를, Hugging Face에 업로드하려면 push_to_hub_gguf를 사용하세요.

지원되는 몇 가지 양자화(quant) 방식은 다음과 같습니다  (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
* `q8_0` -  변환 속도가 빠릅니다. 자원 사용량은 높지만 일반적으로 무난한 수준입니다.
* `q4_k_m` - 추천하는 방식입니다. 어텐션(attention)의 절반과 피드포워드(feed_forward)의 일부 텐서에는 Q6_K를 사용하고, 나머지는 Q4_K를 사용합니다.
* `q5_k_m` - 추천하는 방식입니다. 어텐션의 절반과 피드포워드의 일부 텐서에는 Q6_K를 사용하고, 나머지는 Q5_K를 사용합니다.
"""

# === GGUF Models ===
#
model.push_to_hub_gguf(
    "sssssungjae/qwen-finance-gguf",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
    token=True
)
print("✅ Successfully uploaded GGUF models.")
