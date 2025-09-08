from langchain_openai import ChatOpenAI  # RAGAS와 궁합 좋음  :contentReference[oaicite:8]{index=8}


def build_judge_llm(base_url: str | None, api_key: str, model: str) -> ChatOpenAI:
    """
    채점용 LLM
    - base_url이 None이면 OpenAI 기본 엔드포인트 사용
    - 있으면 내부/대안 엔드포인트로 전환
    """
    return ChatOpenAI(
        base_url=base_url,  # None이면 OpenAI 기본
        api_key=api_key,
        model=model,
        temperature=0,
    )

