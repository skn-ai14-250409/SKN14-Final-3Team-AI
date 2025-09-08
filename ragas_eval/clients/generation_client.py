from typing import List, Dict, Any
from openai import OpenAI  # OpenAI 공식 SDK (vLLM OpenAI-호환에 잘 맞음)  :contentReference[oaicite:6]{index=6}


class GenerationClient:
    """Runpod(vLLM OpenAI-호환) 생성기 래퍼"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)  # base_url만 바꾸면 Runpod로 감  :contentReference[oaicite:7]{index=7}
        self.model = model

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """단일 요청"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", None),
        )
        return resp.choices[0].message.content or ""

    def batch_chat(
        self, messages_list: List[List[Dict[str, Any]]], batch_size: int = 8, **kwargs
    ) -> List[str]:
        """
        간단한 동기 배치 호출(실전엔 asyncio/쓰레드풀로 개선 가능)
        - messages_list를 batch_size씩 끊어서 순차 처리
        """
        outs: List[str] = []
        for i in range(0, len(messages_list), batch_size):
            chunk = messages_list[i : i + batch_size]
            for msgs in chunk:
                outs.append(self.chat(msgs, **kwargs))
        return outs

