"""Public model-communication interfaces."""

from dbagent.llm.chat_client import (
    OpenAIChatCompletionsClient,
    responses_items_to_chat_messages,
)
from dbagent.llm.client import OpenAIResponsesClient
from dbagent.llm.errors import (
    ModelAPIError,
    ModelCommunicationError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelTextualToolMarkupError,
    ModelTimeoutError,
)
from dbagent.llm.models import (
    FunctionCall,
    FunctionTool,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)

__all__ = [
    "FunctionCall",
    "FunctionTool",
    "ModelAPIError",
    "ModelCommunicationError",
    "ModelConfigurationError",
    "ModelConnectionError",
    "ModelProtocolError",
    "ModelRateLimitError",
    "ModelTextualToolMarkupError",
    "ModelRequest",
    "ModelResponse",
    "ModelTimeoutError",
    "OpenAIChatCompletionsClient",
    "OpenAIResponsesClient",
    "responses_items_to_chat_messages",
    "TokenUsage",
]
