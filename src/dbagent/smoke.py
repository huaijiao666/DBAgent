"""Manual, API-consuming smoke test for the model communication layer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dbagent.config import DBAgentConfig
from dbagent.console import safe_print
from dbagent.llm import (
    ModelCommunicationError,
    ModelRequest,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)

_DEFAULT_PROMPT = "Reply with exactly: DBAgent smoke test successful."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send one stateless text request through the configured model API."
    )
    parser.add_argument("prompt", nargs="?", default=_DEFAULT_PROMPT)
    arguments = parser.parse_args(argv)

    try:
        config = DBAgentConfig.from_env()
        client = (
            OpenAIChatCompletionsClient(config)
            if config.api_mode == "chat_completions"
            else OpenAIResponsesClient(config)
        )
        response = client.create_response(ModelRequest(input=arguments.prompt))
    except ModelCommunicationError as error:
        safe_print(f"Smoke test failed: {error}", stream=sys.stderr)
        return 1

    safe_print(f"model: {response.model}")
    safe_print(f"response_id: {response.response_id}")
    safe_print(f"status: {response.status}")
    safe_print(response.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
