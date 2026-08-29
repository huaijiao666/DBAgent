from dataclasses import dataclass


@dataclass(slots=True)
class User:
    user_id: int
    name: str

    def display_name(self) -> str:
        """Return the user-facing name."""
        return self.name.strip()
