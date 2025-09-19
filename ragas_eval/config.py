from dataclasses import dataclass
import os
from dotenv import load_dotenv

# .env 로드 → os.environ에 반영
load_dotenv()

from typing import Any, Dict
_SETTINGS: Dict[str, Any] = {}
try:
    import tomllib  # Python 3.11+
    SETTINGS_FILE = os.environ.get("APP_SETTINGS_FILE", "settings.toml")
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "rb") as f:
            _SETTINGS = tomllib.load(f)
except Exception:
    _SETTINGS = {}

def _get_setting(path: str, default: Any = None) -> Any:
    cur: Any = _SETTINGS
    try:
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur
    except Exception:
        return default

def _env_or(path: str, env_key: str, default: Any) -> Any:
    v = os.environ.get(env_key)
    if isinstance(v, str) and v.strip() == "":
        v = None
    return v if v is not None else _get_setting(path, default)


@dataclass
class AppConfig:
    # Runpod(vLLM) 생성기 엔드포인트 (OpenAI 호환)
    runpod_base_url: str = _env_or("runpod.base_url", "RUNPOD_BASE_URL", "")
    runpod_api_key: str = os.environ.get("RUNPOD_API_KEY", "EMPTY")
    runpod_model: str = _env_or("runpod.model", "RUNPOD_MODEL", "llama3-8b-instruct")

    # Judge LLM (기본: OpenAI GPT-4o mini)
    judge_base_url: str | None = _env_or("judge.base_url", "JUDGE_BASE_URL", None) or None
    judge_api_key: str | None = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    judge_model: str = _env_or("judge.model", "JUDGE_MODEL", "gpt-4o-mini")

    # Dataset
    dataset_name: str = _env_or("dataset.name", "DATASET_NAME", "sssssungjae/finance-kb-mixed-dataset-final")
    dataset_split: str = _env_or("dataset.split", "DATASET_SPLIT", "eval")

    # 컨텍스트 / 배치 / 동시성
    top_k: int = int(_env_or("eval.top_k", "TOP_K", 4))
    gen_batch_size: int = int(_env_or("eval.gen_batch_size", "GEN_BATCH_SIZE", 8))
    max_workers: int = int(_env_or("eval.max_workers", "MAX_WORKERS", 6))

    # 비용(저지 LLM 단가; 필요 시 .env에서 덮어쓰기)
    cost_in_per_m: float = float(_env_or("cost.judge_in_per_m_usd", "JUDGE_COST_IN_USD_PER_M", 0.15))
    cost_out_per_m: float = float(_env_or("cost.judge_out_per_m_usd", "JUDGE_COST_OUT_USD_PER_M", 0.6))

    # 출력
    out_csv: str = _env_or("eval.out_csv", "OUT_CSV", "ragas_detail.csv")

    # 전체 평가(-1) / 일부만 평가(양수)
    max_rows: int = int(_env_or("eval.max_rows", "MAX_ROWS", -1))

    def validate(self) -> None:
        if not self.runpod_base_url:
            raise RuntimeError(
                "RUNPOD_BASE_URL 이 비어있어. .env에 설정해줘 (예: http://<host>:<port>/v1)"
            )
        if not self.judge_api_key:
            raise RuntimeError(
                "Judge API 키가 없어. .env에 OPENAI_API_KEY 또는 JUDGE_API_KEY를 넣어줘."
            )
