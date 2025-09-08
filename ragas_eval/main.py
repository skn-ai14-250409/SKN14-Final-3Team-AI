import argparse
from typing import List
from tqdm import tqdm

from .config import AppConfig
from .data import load_and_parse_dataset, to_ragas_dataset
from .retriever import build_echo_contexts  # 실제 서비스에선 자신만의 retriever로 교체
from .prompting import build_messages_for_chat
from .clients.generation_client import GenerationClient
from .clients.judge_client import build_judge_llm
from .evaluator import run_ragas_eval


def main():
    # ---- CLI 인자 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--max_workers", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true", help="Load data and build messages only; no LLM calls")
    args = ap.parse_args()

    # ---- 설정 로드/덮어쓰기 ----
    cfg = AppConfig()
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.split:
        cfg.dataset_split = args.split
    if args.top_k:
        cfg.top_k = args.top_k
    if args.batch_size:
        cfg.gen_batch_size = args.batch_size
    if args.max_rows is not None:
        cfg.max_rows = args.max_rows
    if args.out_csv:
        cfg.out_csv = args.out_csv
    if args.max_workers:
        cfg.max_workers = args.max_workers
    if not args.dry_run:
        cfg.validate()

    # ---- 데이터 로드 ----
    questions, gts = load_and_parse_dataset(cfg.dataset_name, cfg.dataset_split)
    if cfg.max_rows and cfg.max_rows > 0:
        questions, gts = questions[: cfg.max_rows], gts[: cfg.max_rows]
    print(f"[data] rows={len(questions)}")

    # ---- 컨텍스트 주입 (실전: 벡터DB 리트리버로 교체) ----
    contexts = build_echo_contexts(gts, k=cfg.top_k)

    # ---- 메시지 빌드 ----
    messages_list = build_messages_for_chat(questions, contexts)

    # ---- 생성기(Runpod vLLM OpenAI-호환) 준비 ----
    gen = GenerationClient(cfg.runpod_base_url, cfg.runpod_api_key, cfg.runpod_model)
    if args.dry_run:
        print("[dry-run] dataset rows=", len(questions))
        print("[dry-run] messages built=", len(messages_list))
        if messages_list:
            print("[dry-run] sample message[0]=", messages_list[0])
        return

    # ---- 배치 생성 ----
    answers: List[str] = []
    for i in tqdm(
        range(0, len(messages_list), cfg.gen_batch_size), desc=f"gen bs={cfg.gen_batch_size}"
    ):
        chunk = messages_list[i : i + cfg.gen_batch_size]
        answers.extend(gen.batch_chat(chunk, batch_size=len(chunk)))  # 내부는 순차; 필요 시 비동기화 가능
    print(f"[gen] answers={len(answers)}")

    # ---- RAGAS 입력셋 구성 ----
    ragas_ds = to_ragas_dataset(questions, answers, contexts, gts)

    # ---- Judge LLM 준비 & 평가 ----
    judge = build_judge_llm(cfg.judge_base_url, cfg.judge_api_key, cfg.judge_model)
    out = run_ragas_eval(
        ragas_ds=ragas_ds,
        judge_llm=judge,
        out_csv=cfg.out_csv,
        max_workers=cfg.max_workers,
        show_progress=True,
        cost_in_per_m=cfg.cost_in_per_m,
        cost_out_per_m=cfg.cost_out_per_m,
    )

    # ---- 요약 로그 ----
    print("[summary]", {k: round(v, 3) for k, v in out["summary"].items()})
    if out["usage"] is not None and out["est_cost"] is not None:
        u = out["usage"]
        print(
            f"[tokens] in={u.input_tokens}, out={u.output_tokens} | est_cost=${out['est_cost']:.4f}"
        )


if __name__ == "__main__":
    main()
