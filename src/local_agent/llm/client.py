import httpx
from langchain_openai import ChatOpenAI

from local_agent.config import Settings


class LocalChatOpenAI(ChatOpenAI):
    """保留本地推理服务返回的思考分片，不改变标准工具调用解析。"""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        generation = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if generation is not None and choices:
            delta = choices[0].get("delta") or {}
            if reasoning := delta.get("reasoning_content"):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return generation


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Build a client for vLLM's OpenAI-compatible endpoint."""
    return LocalChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stream_usage=True,
        http_socket_options=(),
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )
