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

# class GGUFModel:
#     def __init__(self, repo_id: str, filename: str, n_ctx: int = 4096, n_threads: int = 8, n_gpu_layers: int = 0):
#         # 캐시 경로 직접 지정 가능 (기본: 현재 폴더 ./models/)
#         local_dir = "./models"
#         os.makedirs(local_dir, exist_ok=True)
#         local_path = os.path.join(local_dir, filename)

#         if os.path.exists(local_path):
#             model_path = local_path
#         else:
#             model_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=local_dir)

#         self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)


# class SLM:
#     def __init__(self):
#         self.gguf = None
#         self.repo_id = "\model\huggingface"
#         self.filename = "sssssungjae/qwen-finance-gguf"

#     def attach_gguf(self, repo_id: str, filename: str, n_ctx: int = 4096, n_threads: int = 8, n_gpu_layers: int = 0):
#         self.gguf = GGUFModel(repo_id, filename, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)

#     def invoke(self, prompt: str) -> str:
#         if isinstance(prompt, list):  
#             # LangChain식 메시지 구조라면 → 문자열로 풀어주기
#             prompt_text = "\n".join([msg.content for msg in prompt])
#         else:
#             prompt_text = str(prompt)


#         output = self.gguf.llm(prompt_text)
#         return output["choices"][0]["text"].strip()  