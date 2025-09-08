from typing import Dict, Any
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from ragas.cost import get_token_usage_for_openai  # OpenAI judge일 때 토큰/비용 집계  :contentReference[oaicite:9]{index=9}


def run_ragas_eval(
    ragas_ds,
    judge_llm,
    out_csv: str,
    max_workers: int = 6,
    show_progress: bool = True,
    cost_in_per_m: float | None = None,
    cost_out_per_m: float | None = None,
) -> Dict[str, Any]:
    """
    RAGAS 4지표 평가 + DF/요약/비용 계산 + CSV 저장
    """
    rc = RunConfig(max_workers=max_workers)

    # judge_llm이 OpenAI 기본일 때만 token_usage_parser 가능(엔드포인트에 따라 미지원일 수 있음)
    token_parser = (
        get_token_usage_for_openai if getattr(judge_llm, "base_url", None) in (None, "",) else None
    )

    result = evaluate(
        dataset=ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        run_config=rc,
        token_usage_parser=token_parser,
        show_progress=show_progress,
        batch_size=8,
    )

    df = result.to_pandas()  # 공식 경로: EvaluationResult → DataFrame  :contentReference[oaicite:10]{index=10}
    df.to_csv(out_csv, index=False)

    # 요약(평균)
    metric_cols = [c for c in df.columns if df[c].dtype.kind in "fi"]
    summary = {k: float(df[k].mean()) for k in metric_cols}

    # 비용(가능할 때만)
    usage, est_cost = None, None
    if token_parser:
        try:
            usage = result.total_tokens()
            cin = (cost_in_per_m or 0.60) / 1e6  # 기본값은 gpt-4o mini 텍스트 단가 예시
            cout = (cost_out_per_m or 2.40) / 1e6
            est_cost = result.total_cost(
                cost_per_input_token=cin, cost_per_output_token=cout
            )
        except Exception:
            pass

    return {"detail_df": df, "summary": summary, "usage": usage, "est_cost": est_cost}

