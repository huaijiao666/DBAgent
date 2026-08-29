"""Manual, API-consuming smoke test for the model communication layer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from forge.config import ForgeConfig
from forge.llm import (
    ModelCommunicationError,
    ModelRequest,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)

_DEFAULT_PROMPT = "Reply with exactly: Forge smoke test successful."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send one stateless text request through the OpenAI Responses API."
    )
    parser.add_argument("prompt", nargs="?", default=_DEFAULT_PROMPT)
    arguments = parser.parse_args(argv)

    try:
        config = ForgeConfig.from_env()
        client = (
            OpenAIChatCompletionsClient(config)
            if config.api_mode == "chat_completions"
            else OpenAIResponsesClient(config)
        )
        response = client.create_response(ModelRequest(input=arguments.prompt))
    except ModelCommunicationError as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1

    print(f"model: {response.model}")
    print(f"response_id: {response.response_id}")
    print(f"status: {response.status}")
    print(response.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
