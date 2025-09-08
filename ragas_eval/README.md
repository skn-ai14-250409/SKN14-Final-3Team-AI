# RAGAS 평가 러너(OpenAI 호환)

OpenAI 호환 LLM 엔드포인트(예: RunPod vLLM, LM Studio, Ollama 등)로 답변을 생성하고, RAGAS 지표로 평가하는 가벼운 파이프라인입니다. 비밀이 아닌 설정은 `settings.toml`, 비밀은 `.env`로 분리되어 있으며, LLM 호출을 생략하는 드라이런 모드도 지원합니다.

간단 사용법은 settings.txt에 있습니다.

## 주요 특징

- OpenAI 호환 생성 클라이언트(RunPod vLLM, LM Studio, Ollama 등과 호환)
- LangChain `ChatOpenAI` 기반 Judge LLM(OpenAI 기본, 또는 OpenAI 호환 서버)
- 지원 지표: faithfulness, answer relevancy, context precision, context recall
- 설정 분리: `settings.toml`(비밀 아님) / `.env`(비밀)
- 1회성 CLI 오버라이드, 드라이런(`--dry_run`)로 빠른 점검

## 저장소 구성

- `settings.toml`: 커밋 가능한 비밀 아님 설정(엔드포인트, 모델명, 데이터셋, 평가 파라미터, 비용 가정)
- `.env`: 비밀만(API 키/토큰). 커밋 금지
- `ragas_detail.csv`: 실행 후 행 단위 결과가 저장되는 CSV
- `run_local.py`: `ragas_eval/` 폴더 안에서 실행할 수 있게 `sys.path`를 보정하는 로컬 실행기
- `requirements.txt`: 파이썬 의존성(Windows에서는 `vllm` 설치 생략)

핵심 패키지 파일
- `config.py`: `.env`(비밀)와 `settings.toml`(비밀 아님)을 로드. 우선순위: CLI > `.env` > `settings.toml` > 기본값
- `main.py`: CLI 엔트리. 데이터셋 로드 → 컨텍스트/메시지 생성 → 답변 생성 → RAGAS 평가
- `data.py`: HF 데이터셋 로드, `(questions, ground_truths)`로 파싱(일반 컬럼 자동 매핑 지원)
- `retriever.py`: 스모크 테스트용 “에코 컨텍스트”(정답을 컨텍스트로 복제) 빌더
- `prompting.py`: 채팅 완료용 system/user 메시지 구성
- `clients/generation_client.py`: OpenAI SDK로 생성 호출(사용자 base URL)
- `clients/judge_client.py`: LangChain `ChatOpenAI`로 Judge 호출
- `evaluator.py`: RAGAS 실행과(가능 시) 토큰/비용 추정

## 요구 사항 및 설치

- Python 3.11+ 권장
- 환경 생성/활성화 및 설치:

```
conda create -n ragas_eval_env python=3.11 -y
conda activate ragas_eval_env
python -m pip install --upgrade pip
pip install -r requirements.txt
```

참고
- Windows에서는 `requirements.txt`가 `vllm` 설치를 건너뜁니다. 원격(OpenAI 호환) 서버만 있으면 로컬 vLLM 설치는 필요 없습니다.

## 설정 방법

비밀 `.env`(커밋 금지)
- `RUNPOD_API_KEY`: RunPod/원격 서버 토큰(필요 시)
- `OPENAI_API_KEY`: OpenAI를 Judge로 쓸 때
- `JUDGE_API_KEY`: OpenAI가 아닌 Judge 엔드포인트에서 필요한 키
- 기타 비밀(HF 토큰, WANDB 등)

비밀 아님 `settings.toml`(커밋 가능)
- `[runpod].base_url`: OpenAI 호환 API Base. 예) `http://localhost:1234/v1`, `https://<runpod-proxy>/v1`
- `[runpod].model`: 서버에서 인식하는 모델 식별자(정확히 일치해야 함)
- `[judge].base_url`, `[judge].model`: Judge LLM 엔드포인트/모델
- `[dataset].name`, `[dataset].split`: HF 데이터셋/스플릿
- `[eval]`: `top_k`, `gen_batch_size`, `max_workers`, `out_csv`, `max_rows`
- `[cost]`: Judge 토큰 단가 가정

설정 파일 경로 변경(선택)
- 환경변수 `APP_SETTINGS_FILE=/path/to/your-settings.toml`

## 빠른 시작(오프라인, LM Studio)

1) LM Studio → Local Server 시작. API Base URL 확인(예: `http://localhost:1234/v1`)
2) 대화형(instruct/chat) 모델을 다운로드/활성화하고, “API identifier”를 그대로 복사(예: `google/gemma-3-1b:2`)
3) `settings.toml` 수정:

```
[runpod]
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-1b:2"  # LM Studio의 “API identifier”와 정확히 일치

[judge]
base_url = "http://localhost:1234/v1"
model    = "google/gemma-3-1b:2"

[dataset]
name  = "sssssungjae/finance-kb-mixed-dataset-final"
split = "eval"

[eval]
max_rows       = 2   # 빠른 스모크
gen_batch_size = 1
max_workers    = 1
```

4) `.env` 최소 비밀(검증 통과용; LM Studio는 보통 Authorization을 무시)

```
RUNPOD_API_KEY=lmstudio
JUDGE_API_KEY=lmstudio
```

5) 실행

- 저장소 루트에서:
  ```
  python -m ragas_eval.main --max_rows 2
  ```
- `ragas_eval/` 폴더에서 보조 실행기 사용:
  ```
  python run_local.py --max_rows 2
  ```

드라이런(LLM 호출 생략)
```
python -m ragas_eval.main --dry_run --max_rows 2
# 또는
python run_local.py --dry_run --max_rows 2
```

## RunPod 실전 튜토리얼

목표: RunPod(vLLM, OpenAI 호환)에서 모델을 서빙하고, 선택한 Judge로 평가하기.

1) RunPod에서 vLLM OpenAI 서버 실행
- 이미지: `ghcr.io/vllm-project/vllm-openai:latest`
- 예시 커맨드:
  ```
  --model <hf_repo_id> \\
  --served-model-name <served_name> \\
  --trust-remote-code
  ```
- 포트 `8000`을 개방하거나 RunPod Proxy 사용(HTTP 엔드포인트 + 토큰). 사설 HF 모델이면 `HUGGING_FACE_HUB_TOKEN` 설정.

2) 서버 확인
- 모델 목록 확인:
  ```
  curl -H "Authorization: Bearer <token>" http://<host>:8000/v1/models
  ```
- 채팅 테스트:
  ```
  curl -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \\
       -X POST http://<host>:8000/v1/chat/completions \\
       -d '{"model":"<served_name>","messages":[{"role":"user","content":"hello"}]}'
  ```

3) 이 저장소 설정 업데이트
- `settings.toml`:
  ```
  [runpod]
  base_url = "https://<runpod-proxy-or-host>/v1"
  model    = "<served_name>"   # 반드시 --served-model-name와 동일

  # Judge 선택
  # (A) OpenAI Judge(기본)
  [judge]
  base_url = null
  model    = "gpt-4o-mini"      # 임의의 OpenAI 모델

  # (B) Judge도 RunPod로
  # [judge]
  # base_url = "https://<runpod-proxy-or-host>/v1"
  # model    = "<served_name_or_other>"

  [dataset]
  name  = "<hf_user>/<dataset>"
  split = "eval"
  ```
- `.env`:
  ```
  RUNPOD_API_KEY=<runpod_proxy_token_or_api_key>
  # OpenAI Judge일 때
  OPENAI_API_KEY=<openai_key>
  # Judge도 RunPod일 때
  JUDGE_API_KEY=<동일 또는 별도 토큰>
  ```

4) 실행
```
python -m ragas_eval.main --max_rows 50
```

## 데이터셋

`settings.toml` 기본값:
- `dataset.name = "sssssungjae/finance-kb-mixed-dataset-final"`
- `dataset.split = "eval"`

로더 동작(`data.py`)
- 단일 `text` 컬럼(Qwen 스타일 전사)이면 `<|im_start|>user`/`assistant` 블록에서 Q/A를 파싱
- 그렇지 않으면 질문/정답 컬럼을 자동 탐색: `question|query|user_input|q` vs. `ground_truth|reference|answer|a`
- 스키마가 다르면 `load_and_parse_dataset` 안에 간단한 분기 추가로 매핑:

```
# load_and_parse_dataset(...) 내부
q_col, a_col = "my_q", "my_a"
if q_col in ds.column_names and a_col in ds.column_names:
    ds2 = ds.filter(lambda r: isinstance(r[q_col], str) and r[q_col].strip() and isinstance(r[a_col], str) and r[a_col].strip())
    return ds2[q_col], ds2[a_col]
```

파일 수정 없이 1회성 오버라이드
```
python -m ragas_eval.main --dataset_name "<hf_user>/<dataset>" --split "eval"
```

## 평가 파라미터(`[eval]` in settings.toml)
- `top_k`: 질문당 컨텍스트 조각 수(스모크에서는 에코 컨텍스트)
- `gen_batch_size`: 생성 배치 크기
- `max_workers`: 평가 병렬성
- `out_csv`: 행 단위 결과 CSV 경로
- `max_rows`: 처리할 최대 행 수(전체는 -1)

## 비용 추정
- Judge가 OpenAI(또는 OpenAI 사용량 응답을 제공하는 서버)인 경우 `evaluator.py`가 `[cost]` 값으로 토큰/비용을 추정할 수 있습니다.

## 폴더별 실행
- 권장: 저장소 루트에서 `python -m ragas_eval.main ...`
- `ragas_eval/`에서 실행하려면 보조 실행기 사용: `python run_local.py ...`
- `run_local.py`는 런타임에 `sys.path`를 보정하므로 IDE가 임포트를 노란 줄로 표시할 수 있습니다(정상).

## 트러블슈팅
- `model_not_found`: `settings.toml`의 모델명이 서버의 served name과 다름. 서버/LM Studio의 “API identifier”를 그대로 복사해 사용하세요.
- `401 Unauthorized`: `.env`의 토큰(`RUNPOD_API_KEY`/`JUDGE_API_KEY`) 확인, curl 테스트 시 헤더 포함
- 연결 오류: `base_url`에 `/v1` 포함, 포트 개방/프록시 설정/방화벽 허용 확인
- Judge 단계 `IndexError(list index out of range)`: 작은/비지시형 모델이 빈 응답을 반환할 수 있음. 지시형 모델로 교체하고 `max_workers`를 줄이세요.
- `Unexpected endpoint or method (GET /v1/chat/completions)`: 이 엔드포인트는 `POST`만 허용
- Windows + vLLM: 로컬 `vllm` 설치 대신 원격 서버(RunPod) 또는 LM Studio/Ollama 사용 권장

## 라이선스
- 필요 시 원하는 라이선스를 이 섹션에 추가하세요.

