import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests


class ChatClient:
    """Small adapter matching the existing pipeline's get_model_response API."""

    def __init__(self, model_id: str, max_workers: int = 4, max_retries: int = 3):
        self.model_id = model_id
        self.max_workers = max_workers
        self.max_retries = max_retries

    def get_model_response(
        self,
        system_prompt: str | list[str],
        user_prompt: str | list[str],
        model_id: str | None = None,
        max_new_tokens: int = 2000,
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> str | list[str]:
        if isinstance(system_prompt, list) and isinstance(user_prompt, list):
            if len(system_prompt) != len(user_prompt):
                raise ValueError("system_prompt and user_prompt lists must match")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._call,
                        system,
                        user,
                        model_id or self.model_id,
                        max_new_tokens,
                        temperature,
                        **kwargs,
                    )
                    for system, user in zip(system_prompt, user_prompt)
                ]
            return [future.result() for future in futures]

        if isinstance(system_prompt, str) and isinstance(user_prompt, str):
            return self._call(
                system_prompt,
                user_prompt,
                model_id or self.model_id,
                max_new_tokens,
                temperature,
                **kwargs,
            )

        raise ValueError("system_prompt and user_prompt must both be strings or lists")

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class OllamaClient(ChatClient):
    def __init__(
        self,
        model_id: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "top_p": kwargs.get("top_p", 0.9),
            },
        }
        return _post_chat(f"{self.base_url}/api/chat", body, self.max_retries)


class OpenAICompatClient(ChatClient):
    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "not-needed"

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return _post_chat(
            f"{self.base_url}/chat/completions",
            body,
            self.max_retries,
            headers=headers,
        )


class RITSClient(OpenAICompatClient):
    def __init__(self, model_id: str, **kwargs: Any):
        super().__init__(
            model_id=model_id,
            base_url=kwargs.pop("base_url", os.getenv("RITS_BASE_URL", "")),
            api_key=kwargs.pop("api_key", os.getenv("RITS_API_KEY", "")),
            **kwargs,
        )


class ModelClientFactory:
    @staticmethod
    def create_client(
        backend: str = "ollama",
        model_id: str = "qwen2.5-coder:7b",
        **kwargs: Any,
    ) -> ChatClient:
        normalized = backend.lower().replace("_", "-")
        if normalized == "ollama":
            return OllamaClient(model_id=model_id, **kwargs)
        if normalized in {"openai", "openai-compatible"}:
            return OpenAICompatClient(model_id=model_id, **kwargs)
        if normalized == "rits":
            return RITSClient(model_id=model_id, **kwargs)
        raise ValueError("Unknown backend. Use ollama, openai-compatible, or rits.")


def _post_chat(
    url: str,
    body: dict[str, Any],
    max_retries: int,
    headers: dict[str, str] | None = None,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=body, headers=headers, timeout=180)
            response.raise_for_status()
            payload = response.json()
            if "message" in payload:
                return payload["message"]["content"].strip()
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Model request failed after {max_retries} attempts: {exc}"
                ) from exc
            time.sleep(2**attempt)
