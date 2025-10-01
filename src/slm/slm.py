# from typing import List, Any, Dict

# from langchain_openai import ChatOpenAI

# from src.config import MODEL_KEY, MODEL_NAME

# class SLM:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#                 model=MODEL_NAME,
#                 api_key=MODEL_KEY,
#                 temperature=0.1,  # 일관성 향상 및 속도 개선
#                 max_tokens=500,   # 응답 길이 더 제한 (1000 → 500)
#                 request_timeout=30,  # 타임아웃 증가 (10 → 30)
#                 max_retries=3  # 재시도 횟수 증가 (1 → 3)
#             )
    
#     def invoke(
#         self,
#         prompt
#     ) -> str:
#         response = self.llm.invoke(prompt)
#         return response.content
    
#     def get_structured_output(
#         self,
#         prompt,
#         output_structure_cls
#     ):
#         structured_llm = self.llm.with_structured_output(
#             output_structure_cls
#         )
#         response = structured_llm.invoke(prompt)
#         return response


from typing import Any
import logging
import requests

from langchain_openai import ChatOpenAI

from src.config import (
    ENABLE_OPENAI_FALLBACK,
    MODEL_BASE_URL,
    MODEL_KEY,
    MODEL_NAME,
    MODEL_PROVIDER,
    FALLBACK_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class _FallbackRunnable:
    """Wraps a runnable so that it retries with fallback LLM on failure."""

    def __init__(self, primary, fallback, fallback_label: str):
        self._primary = primary
        self._fallback = fallback
        self._fallback_label = fallback_label

    def _call(self, method_name: str, *args, **kwargs):
        method = getattr(self._primary, method_name, None)
        fallback_method = getattr(self._fallback, method_name, None) if self._fallback else None

        if method is None:
            if fallback_method is None:
                raise AttributeError(f"Method '{method_name}' is not available on primary or fallback runnable")
            logger.debug("Primary runnable lacks '%s', using fallback directly", method_name)
            return fallback_method(*args, **kwargs)

        try:
            return method(*args, **kwargs)
        except Exception as primary_error:
            if not fallback_method:
                raise
            logger.warning(
                "vLLM 호출 실패 – OpenAI 폴백 모델 '%s'로 재시도합니다: %s",
                self._fallback_label,
                primary_error,
            )
            try:
                return fallback_method(*args, **kwargs)
            except Exception as fallback_error:  # pragma: no cover - 재시도도 실패한 경우
                fallback_error.__cause__ = primary_error
                raise fallback_error

    async def _call_async(self, method_name: str, *args, **kwargs):
        method = getattr(self._primary, method_name, None)
        fallback_method = getattr(self._fallback, method_name, None) if self._fallback else None

        if method is None:
            if fallback_method is None:
                raise AttributeError(f"Async method '{method_name}' is not available on primary or fallback runnable")
            logger.debug("Primary runnable lacks async '%s', using fallback directly", method_name)
            return await fallback_method(*args, **kwargs)

        try:
            return await method(*args, **kwargs)
        except Exception as primary_error:
            if not fallback_method:
                raise
            logger.warning(
                "vLLM 호출 실패 – OpenAI 폴백 모델 '%s'로 재시도합니다: %s",
                self._fallback_label,
                primary_error,
            )
            try:
                return await fallback_method(*args, **kwargs)
            except Exception as fallback_error:  # pragma: no cover
                fallback_error.__cause__ = primary_error
                raise fallback_error

    def invoke(self, *args, **kwargs):
        return self._call("invoke", *args, **kwargs)

    def batch(self, *args, **kwargs):
        return self._call("batch", *args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._call("stream", *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self._call_async("ainvoke", *args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self._call_async("abatch", *args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self._call_async("astream", *args, **kwargs)


class _FallbackChatWrapper:
    """ChatOpenAI 호환 래퍼로, 실패 시 자동으로 OpenAI 모델로 폴백한다."""

    def __init__(self, primary: ChatOpenAI, fallback: ChatOpenAI | None, fallback_label: str):
        self._primary = primary
        self._fallback = fallback
        self._fallback_label = fallback_label

    def _call(self, method_name: str, *args, **kwargs):
        method = getattr(self._primary, method_name, None)
        fallback_method = getattr(self._fallback, method_name, None) if self._fallback else None

        if method is None:
            if fallback_method is None:
                raise AttributeError(f"Method '{method_name}' is not available on primary or fallback ChatOpenAI")
            logger.debug("Primary ChatOpenAI lacks '%s', using fallback directly", method_name)
            return fallback_method(*args, **kwargs)

        try:
            return method(*args, **kwargs)
        except Exception as primary_error:
            if not fallback_method:
                raise
            logger.warning(
                "vLLM 호출 실패 – OpenAI 폴백 모델 '%s'로 재시도합니다: %s",
                self._fallback_label,
                primary_error,
            )
            try:
                return fallback_method(*args, **kwargs)
            except Exception as fallback_error:  # pragma: no cover
                fallback_error.__cause__ = primary_error
                raise fallback_error

    async def _call_async(self, method_name: str, *args, **kwargs):
        method = getattr(self._primary, method_name, None)
        fallback_method = getattr(self._fallback, method_name, None) if self._fallback else None

        if method is None:
            if fallback_method is None:
                raise AttributeError(f"Async method '{method_name}' is not available on primary or fallback ChatOpenAI")
            logger.debug("Primary ChatOpenAI lacks async '%s', using fallback directly", method_name)
            return await fallback_method(*args, **kwargs)

        try:
            return await method(*args, **kwargs)
        except Exception as primary_error:
            if not fallback_method:
                raise
            logger.warning(
                "vLLM 호출 실패 – OpenAI 폴백 모델 '%s'로 재시도합니다: %s",
                self._fallback_label,
                primary_error,
            )
            try:
                return await fallback_method(*args, **kwargs)
            except Exception as fallback_error:  # pragma: no cover
                fallback_error.__cause__ = primary_error
                raise fallback_error

    def invoke(self, *args, **kwargs):
        return self._call("invoke", *args, **kwargs)

    def batch(self, *args, **kwargs):
        return self._call("batch", *args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._call("stream", *args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self._call_async("ainvoke", *args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self._call_async("abatch", *args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self._call_async("astream", *args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        primary_structured = self._primary.with_structured_output(*args, **kwargs)
        fallback_structured = (
            self._fallback.with_structured_output(*args, **kwargs)
            if self._fallback
            else None
        )
        return _FallbackRunnable(primary_structured, fallback_structured, self._fallback_label)

    def bind_tools(self, *args, **kwargs):
        primary_bound = self._primary.bind_tools(*args, **kwargs)
        fallback_bound = (
            self._fallback.bind_tools(*args, **kwargs)
            if self._fallback
            else None
        )
        return _FallbackRunnable(primary_bound, fallback_bound, self._fallback_label)

    def __getattr__(self, item):
        return getattr(self._primary, item)


class SLM:
    def __init__(self):
        self.provider = MODEL_PROVIDER
        self._vllm_available = None  # vLLM 상태 캐싱
        
        # vLLM 서버가 실제로 실행 중인지 먼저 확인
        vllm_available = (
            MODEL_PROVIDER == "vllm"
            and MODEL_BASE_URL
            and self._check_vllm_server()

        )
        
        # vLLM 사용 가능 여부에 따라 primary_llm 생성
        if vllm_available:
            primary_llm = self._create_llm(
                model_name=MODEL_NAME,
                api_key=MODEL_KEY,
                base_url=MODEL_BASE_URL,
                allow_dummy_key=True,  # vLLM은 API 키 불필요
            )
        else:
            # vLLM 사용 불가 시 OpenAI로 직접 생성
            primary_llm = self._create_llm(
                model_name=FALLBACK_MODEL_NAME or MODEL_NAME,
                api_key=MODEL_KEY,
                base_url=None,
                allow_dummy_key=False,
            )
            logger.info("vLLM 사용 불가 - OpenAI 모델 '%s'로 직접 사용", FALLBACK_MODEL_NAME or MODEL_NAME)

        fallback_llm: ChatOpenAI | None = None
        
        # vLLM 사용 가능 시에만 폴백 설정
        if (
            vllm_available
            and MODEL_KEY
            and ENABLE_OPENAI_FALLBACK
            and FALLBACK_MODEL_NAME
        ):
            try:
                fallback_llm = self._create_llm(
                    model_name=FALLBACK_MODEL_NAME,
                    api_key=MODEL_KEY,
                    base_url=None,
                    allow_dummy_key=False,
                )
                logger.info(
                    "vLLM 기본 설정 – OpenAI 모델 '%s'로 자동 폴백 활성화",
                    FALLBACK_MODEL_NAME,
                )
            except Exception:
                logger.warning(
                    "OpenAI 폴백 LLM 초기화에 실패했습니다. vLLM 응답 실패 시 예외가 그대로 전달됩니다.",
                    exc_info=True,
                )

        # LLM 설정
        fallback_label = FALLBACK_MODEL_NAME or "openai-fallback"
        if fallback_llm:
            self.llm = _FallbackChatWrapper(primary_llm, fallback_llm, fallback_label)
        else:
            self.llm = primary_llm
        
        # 내부 참조 저장
        self._primary_llm = primary_llm
        self._fallback_llm = fallback_llm
    
    def _check_vllm_server(self) -> bool:
        """vLLM 서버가 실제로 실행 중인지 확인 (캐싱)"""
        if self._vllm_available is not None:
            return self._vllm_available
            
        if not MODEL_BASE_URL:
            self._vllm_available = False
            return False
        
        try:
            # vLLM 서버 헬스체크
            response = requests.get(f"{MODEL_BASE_URL}/health", timeout=2)
            self._vllm_available = response.status_code == 200
            if not self._vllm_available:
                logger.warning("vLLM 서버 연결 실패 - OpenAI 모델로 전환")
            return self._vllm_available
        except Exception:
            logger.warning("vLLM 서버 연결 실패 - OpenAI 모델로 전환")
            self._vllm_available = False
            return False

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
        structured_llm = self.llm.with_structured_output(output_structure_cls)
        return structured_llm.invoke(prompt)

    def _create_llm(
        self,
        *,
        model_name: str,
        api_key: str | None,
        base_url: str | None,
        allow_dummy_key: bool,
    ) -> ChatOpenAI:
        llm_kwargs = dict(
            model=model_name,
            temperature=0.1,  # 일관성 향상 및 속도 개선
            max_tokens=500,   # 응답 길이 더 제한 (1000 → 500)
            request_timeout=15,  # 타임아웃 감소 (30 → 15) - 더 빠른 응답
            max_retries=1,  # 재시도 횟수 감소 (3 → 1) - 빠른 실패
        )
        if base_url:
            llm_kwargs["base_url"] = base_url

        if api_key:
            llm_kwargs["api_key"] = api_key
        elif allow_dummy_key:
            # vLLM과 같은 OpenAI 호환 서버는 API 키를 요구하지 않을 수 있다.
            llm_kwargs["api_key"] = "dummy-key"
        else:
            llm_kwargs["api_key"] = "EMPTY"

        return ChatOpenAI(**llm_kwargs)