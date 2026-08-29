class BaseService:
    """Common service behavior."""

    @property
    def service_name(self) -> str:
        return type(self).__name__
