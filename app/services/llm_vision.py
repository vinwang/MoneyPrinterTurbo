"""OpenAI-compatible multimodal requests used by local asset indexing."""

from __future__ import annotations

import base64
from typing import Any, Mapping, Sequence

from openai import AzureOpenAI, OpenAI

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider

_VISION_MAX_OUTPUT_TOKENS = 2048
_DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


def _validate_input(prompt: str, image_bytes: Sequence[bytes]) -> None:
    """校验视觉请求文本和代表帧字节。"""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("vision prompt is required")
    if not isinstance(image_bytes, Sequence) or not image_bytes:
        raise ValueError("vision analysis requires at least one image")
    if any(
        not isinstance(payload, (bytes, bytearray, memoryview)) or not payload
        for payload in image_bytes
    ):
        raise ValueError("vision images must be non-empty binary data")


def _resolve_context(app_config: Mapping | None) -> tuple[Any, str, str, str, str, Mapping]:
    """解析 Provider、模型、鉴权和服务地址，并拒绝不支持的模型适配器。"""
    runtime_config = app_config if app_config is not None else config.app
    provider_id = str(
        runtime_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
    ).lower()
    provider = get_llm_provider(provider_id)
    if provider is None:
        raise ValueError(f"{provider_id}: unsupported llm provider")
    if provider.adapter not in {
        "openai_compatible",
        "azure",
        "cloudflare_ai_gateway",
        "modelscope",
        "gemini",
    }:
        raise ValueError(
            f"{provider_id}: visual indexing requires an OpenAI-compatible vision model"
        )
    api_key = runtime_config.get(provider.config_key("api_key"), "")
    configured_model = runtime_config.get(provider.config_key("model_name"), "")
    model_name = provider.resolve_model_name(configured_model)
    configured_base_url = runtime_config.get(provider.config_key("base_url"), "")
    base_url = provider.resolve_base_url(configured_base_url)
    if provider_id == "ollama":
        api_key = "ollama"
        if not base_url:
            base_url = config.get_default_ollama_base_url()
    if provider.requires_api_key and not api_key:
        raise ValueError(f"{provider_id}: api_key is not set")
    if provider.requires_model_name and not model_name:
        raise ValueError(f"{provider_id}: model_name is not set")
    if provider.requires_base_url and not base_url:
        raise ValueError(f"{provider_id}: base_url is not set")
    return provider, provider_id, api_key, model_name, base_url, runtime_config


def _openai_messages(prompt: str, image_bytes: Sequence[bytes]) -> list[dict[str, Any]]:
    """将代表帧编码成 OpenAI Chat Completions 的多模态消息。"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for payload in image_bytes:
        encoded = base64.b64encode(bytes(payload)).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return [{"role": "user", "content": content}]


def _request_gemini(
    prompt: str,
    image_bytes: Sequence[bytes],
    *,
    api_key: str,
    model_name: str,
    base_url: str,
    provider_id: str,
) -> str:
    """通过 Gemini SDK 提交多模态视觉请求。"""
    from google import genai
    from google.genai import types
    from app.services import llm

    http_options = types.HttpOptions(base_url=base_url) if base_url else None
    contents = [prompt]
    contents.extend(
        types.Part.from_bytes(data=bytes(payload), mime_type="image/jpeg")
        for payload in image_bytes
    )
    with genai.Client(api_key=api_key, http_options=http_options) as client:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                        max_output_tokens=_VISION_MAX_OUTPUT_TOKENS,
            ),
        )
    return llm._normalize_text_response(response.text, provider_id)


def _openai_client(
    provider,
    *,
    provider_id: str,
    api_key: str,
    base_url: str,
    runtime_config,
):
    """创建普通、Azure 或 Cloudflare OpenAI SDK 客户端。"""
    if provider_id == "cloudflare":
        account_id = runtime_config.get(provider.config_key("account_id"), "")
        gateway_id = runtime_config.get(provider.config_key("gateway_id"), "default")
        return OpenAI(
            api_key=api_key,
            base_url=(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
            ),
            default_headers={"cf-aig-gateway-id": gateway_id},
        )
    if provider.adapter == "azure":
        api_version = runtime_config.get(
            provider.config_key("api_version"), _DEFAULT_AZURE_API_VERSION
        )
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_vision_response(
    prompt: str,
    image_bytes: Sequence[bytes],
    app_config: Mapping | None = None,
) -> str:
    """
    使用当前 OpenAI-compatible Provider 分析压缩代表帧。

    @param prompt 视觉分析任务说明，不包含原始视频路径。
    @param image_bytes 已压缩的代表帧字节序列。
    @param app_config 可选的提交时 LLM 配置快照。
    @returns 模型原始文本；失败时返回脱敏的 `Error: ...` 文本。
    """
    from app.services import llm
    try:
        _validate_input(prompt, image_bytes)
        provider, provider_id, api_key, model_name, base_url, runtime_config = (
            _resolve_context(app_config)
        )
        if provider.adapter == "gemini":
            return _request_gemini(
                prompt,
                image_bytes,
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                provider_id=provider_id,
            )
        client = _openai_client(
            provider,
            provider_id=provider_id,
            api_key=api_key,
            base_url=base_url,
            runtime_config=runtime_config,
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=_openai_messages(prompt, image_bytes),
        )
        return llm._extract_chat_completion_text(response, provider_id)
    except Exception as exc:
        return f"Error: {llm._sanitize_error_message(exc)}"
