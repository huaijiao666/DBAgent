from .base import BaseService
from .models import User
from .repository import UserRepository, normalize_id


class UserService(BaseService):
    def __init__(self, repository: UserRepository) -> None:
        self.repository = repository

    def find(self, raw_id: str) -> User | None:
        user_id = normalize_id(raw_id)
        return self.repository.get(user_id)
