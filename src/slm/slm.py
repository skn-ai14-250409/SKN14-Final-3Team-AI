from typing import List, Any, Dict

from langchain_openai import ChatOpenAI

from src.config import MODEL_KEY, MODEL_NAME

class SLM:
    def __init__(self):
        self.llm = ChatOpenAI(
                model=MODEL_NAME,
                api_key=MODEL_KEY
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
    ) -> Dict[str, Any]:
        structured_llm = self.llm.with_structured_output(
            output_structure_cls
        )
        response = structured_llm.invoke(prompt)
        return response