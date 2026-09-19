from abc import ABC, abstractmethod
from typing import Any, TypeAlias


class BaseMaterializer(ABC):
    @abstractmethod
    def supports(self, value: Any) -> bool:
        pass

    @abstractmethod
    @property
    def s3_support(self) -> bool:
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(self, value, *args, **kwars):
        pass
