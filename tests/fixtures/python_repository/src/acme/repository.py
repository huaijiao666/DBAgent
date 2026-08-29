from .models import User


class UserRepository:
    def get(self, user_id: int) -> User | None:
        if user_id <= 0:
            return None
        return User(user_id=user_id, name="Example")


def normalize_id(raw: str) -> int:
    return int(raw.strip())
