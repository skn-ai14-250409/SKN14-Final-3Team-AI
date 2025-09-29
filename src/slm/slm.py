from typing import List, Any, Dict

from langchain_openai import ChatOpenAI

from src.config import MODEL_KEY, MODEL_NAME

class SLM:
    def __init__(self):
        self.llm = ChatOpenAI(
                model=MODEL_NAME,
                api_key=MODEL_KEY,
                temperature=0.1,  # 일관성 향상 및 속도 개선
                max_tokens=500,   # 응답 길이 더 제한 (1000 → 500)
                request_timeout=10,  # 타임아웃 더 짧게 (15 → 10)
                max_retries=1  # 재시도 횟수 더 제한 (2 → 1)
            )
    
    def invoke(
        self,
        prompt
    ) -> str:
        response = self.llm.invoke(prompt)
        return response.content
    
    def get_structured_output(
        self,
        prompt,
        output_structure_cls
    ):
        structured_llm = self.llm.with_structured_output(
            output_structure_cls
        )
        response = structured_llm.invoke(prompt)
        return response
