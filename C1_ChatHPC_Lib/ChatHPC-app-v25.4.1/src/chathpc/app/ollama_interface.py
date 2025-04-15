from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chathpc.app.app import AppConfig

from ollama import ChatResponse, GenerateResponse, chat, generate

from chathpc.app.utils import template_utils


def ollama_evaluate(config: AppConfig, model_name: str, **kwargs) -> str | None:
    """Evaluate a prompt using Ollama's generate API.

    Args:
        model_name (str): Name of the Ollama model to use
        **kwargs: Keyword arguments containing prompt and context
            - prompt: The input prompt to evaluate
            - context: System context/instructions

    Returns:
        str | None: The generated response text, stripped of whitespace,
                    or None if generation fails

    Example:
        response = ollama_evaluate("llama2",
                                    prompt="What is 2+2?",
                                    context="You are a math tutor")
    """

    kw = template_utils.map_keywords(kwargs)

    response: GenerateResponse = generate(
        model=model_name,
        prompt=kw["prompt"],
        system=kw["context"],
        options={"temperature": 0.0, "num_predict": config.max_response_tokens},
    )
    return response.response.strip()


def ollama_chat_evaluate(config: AppConfig, model_name: str, **kwargs) -> str | None:
    """Evaluate a prompt using Ollama's chat API.

    Args:
        model_name (str): Name of the Ollama model to use
        **kwargs: Keyword arguments containing prompt and context
            - prompt: The input prompt to evaluate
            - context: System context/instructions

    Returns:
        str | None: The generated chat response content, stripped of whitespace,
                    or None if chat fails

    Example:
        response = ollama_chat_evaluate("llama2",
                                        prompt="What is 2+2?",
                                        context="You are a math tutor")
    """

    kw = template_utils.map_keywords(kwargs)

    response: ChatResponse = chat(
        model=model_name,
        options={"temperature": 0.0, "num_predict": config.max_response_tokens},
        messages=[
            {"role": "system", "content": kw["context"]},
            {"role": "user", "content": kw["prompt"]},
        ],
    )
    return response.message.content
