from typing import List, Dict, Any


def build_messages_for_chat(questions: List[str], contexts: List[List[str]]) -> List[List[Dict[str, Any]]]:
    """
    OpenAI-호환 Chat 메시지 생성
    - [system, user] 구조
    - user에 컨텍스트와 질문을 포함
    """
    messages_all = []
    for q, ctx in zip(questions, contexts):
        ctx_block = "\n".join([f"- {c}" for c in ctx]) if ctx else "N/A"
        user_text = (
            "다음 질문에 금융 도메인 지식을 바탕으로 한국어로 간결히 답해줘.\n"
            "가능하면 제공된 컨텍스트만 근거로 사용하고, 없으면 모른다고 말해.\n\n"
            f"[컨텍스트]\n{ctx_block}\n\n[질문]\n{q}"
        )
        messages_all.append([
            {"role": "system", "content": "You are a helpful banking/credit analysis assistant."},
            {"role": "user", "content": user_text},
        ])
    return messages_all

