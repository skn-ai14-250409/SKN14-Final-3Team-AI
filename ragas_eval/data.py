from typing import List, Tuple
from datasets import load_dataset, Dataset  # HF 공식  :contentReference[oaicite:5]{index=5}


def load_and_parse_dataset(dataset_name: str, split: str) -> Tuple[List[str], List[str]]:
    """
    다양한 스키마를 자동 감지해서 (questions, ground_truths) 반환
    - Qwen-style(text) 또는 (question/reference|ground_truth) 조합 지원
    """
    ds = load_dataset(dataset_name, split=split)

    if "text" in ds.column_names:
        def _parse_text(row):
            t = row.get("text", "")
            q, a = None, None
            try:
                q = t.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
                a = t.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
            except Exception:
                pass
            return {"_q": q, "_gt": a}

        ds2 = ds.map(_parse_text)
        ds2 = ds2.filter(lambda r: bool(r["_q"]) and bool(r["_gt"]))
        return ds2["_q"], ds2["_gt"]

    cand_q = [c for c in ["question", "query", "user_input", "q"] if c in ds.column_names]
    cand_a = [c for c in ["ground_truth", "reference", "answer", "a"] if c in ds.column_names]
    if cand_q and cand_a:
        q_col, a_col = cand_q[0], cand_a[0]
        ds2 = ds.filter(
            lambda r: isinstance(r[q_col], str)
            and r[q_col].strip() != ""
            and isinstance(r[a_col], str)
            and r[a_col].strip() != ""
        )
        return ds2[q_col], ds2[a_col]

    raise ValueError(f"지원되지 않는 데이터 포맷: {ds.column_names}")


def to_ragas_dataset(
    questions: List[str], answers: List[str], contexts: List[List[str]], gts: List[str]
) -> Dataset:
    """RAGAS 요구 스키마로 변환"""
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,  # List[List[str]]
        "ground_truth": gts,
    })

