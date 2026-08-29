"""Exceptions exposed by the model-communication boundary."""

from __future__ import annotations


class ModelCommunicationError(RuntimeError):
    """Base class for failures while communicating with a model provider."""


class ModelConfigurationError(ModelCommunicationError):
    """Raised when model communication is configured incorrectly."""


class ModelTimeoutError(ModelCommunicationError):
    """Raised after an OpenAI request exceeds its timeout and retries."""


class ModelConnectionError(ModelCommunicationError):
    """Raised when the OpenAI API cannot be reached."""


class ModelAPIError(ModelCommunicationError):
    """Raised when OpenAI rejects a request or returns a failed response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        response_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response_id = response_id


class ModelRateLimitError(ModelAPIError):
    """Raised when OpenAI returns HTTP 429 after SDK retries."""


class ModelProtocolError(ModelCommunicationError):
    """Raised when an SDK response does not match the expected contract."""
