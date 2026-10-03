"""`DefaultMaterializer`: the catch-all that refuses to serialize."""

from typing import Any

from .base_materializer import BaseMaterializer


class DefaultMaterializer(BaseMaterializer):
    """Fallback materializer used when no specialised `BaseMaterializer`
    supports a task input/output's type.

    Deliberately refuses to (de)serialize rather than silently falling back
    to an unsafe generic mechanism such as pickle; register a custom
    `BaseMaterializer` subclass for types that need one. Since `_save`
    always raises, no metadata sidecar is ever written for it either -
    there is nothing to record a successful save of.
    """

    name = "default"

    @classmethod
    def supports_type(cls, type_hint: Any) -> bool:
        """Always reports support, since this is the catch-all fallback.

        Args:
            type_hint: The declared type to check.

        Returns:
            Always `True`.
        """

        return True

    def supports(self, value: Any) -> bool:
        """Always reports support, since this is the catch-all fallback.

        Args:
            value: The value to check.

        Returns:
            Always `True`.
        """

        return True

    @property
    def s3_support(self) -> bool:
        """Whether this materializer can hand its output directly to the
        `ArtifactStore`.

        Returns:
            Always `False`.
        """

        return False

    def _load(self, path: str) -> Any:
        """Always raises: there is no safe, generic way to deserialize an
        arbitrary type.

        Args:
            path: Unused.

        Raises:
            NotImplementedError: Always.
        """

        raise NotImplementedError(
            "No materializer is registered for this type. Register a "
            "custom BaseMaterializer subclass for it rather than relying on "
            "an unsafe fallback such as pickle."
        )

    def _save(self, value: Any, path: str) -> None:
        """Always raises: there is no safe, generic way to serialize an
        arbitrary type.

        Args:
            value: Unused.
            path: Unused.

        Raises:
            NotImplementedError: Always.
        """

        raise NotImplementedError(
            "No materializer is registered for this type. Register a "
            "custom BaseMaterializer subclass for it rather than relying on "
            "an unsafe fallback such as pickle."
        )
