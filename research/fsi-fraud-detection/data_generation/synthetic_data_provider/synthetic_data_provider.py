"""Base abstractions for synthetic data providers.

Defines the generic interface that all synthetic data providers must implement,
along with the protocol and type alias used to describe per-attribute value
generator callables.
"""

from abc import ABC, abstractmethod


class SyntheticDataProvider[T](ABC):
    """Abstract base class for synthetic data providers.

    Subclasses wrap a concrete data source (e.g. Faker, an RNG) and expose it
    through `provide`, returning domain-specific synthetic data of type `T`.
    """

    def __init__(self): ...

    @abstractmethod
    def provide(self) -> T:
        """Return a single synthetic data sample of type `T`."""
        raise NotImplementedError("Subclasses must implement the provide method.")
