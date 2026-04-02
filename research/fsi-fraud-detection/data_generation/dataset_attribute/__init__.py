"""Dataset attribute descriptors for synthetic payment data.

``PaymentDatasetAttribute`` represents a single column and
``PaymentDatasetAttributeGroup`` represents a group of columns produced
together (e.g. latitude + longitude).  Both wrap an
``AttributeDataProviderProtocol`` callable that generates column values
for a batch of rows held in a ``pd.DataFrame``.
"""

import pandas as pd
from data_generation.synthetic_data_provider.synthetic_data_provider import (
    SyntheticDataProvider,
)
from data_generation.attribute_data_provider import AttributeDataProviderProtocol


class PaymentDatasetAttributeGroup[T: SyntheticDataProvider]:
    """A named group of one or more columns produced by a single provider callable.

    Args:
        names: Tuple or list of column names (will be uppercased).
        attribute_data_provider: Callable conforming to
            ``AttributeDataProviderProtocol`` that generates the column(s).
    """

    def __init__(
        self,
        names: tuple[str, ...] | list[str],
        attribute_data_provider: AttributeDataProviderProtocol[T],
    ):
        assert len(names) > 0, "At least one attribute name must be provided"
        assert all(name is not None and name.strip() != "" for name in names), (
            "All attribute names must be non-empty strings"
        )
        self.names = tuple(name.upper() for name in names)
        self.attribute_data_provider = attribute_data_provider

    def __eq__(self, other):
        if not isinstance(other, PaymentDatasetAttributeGroup):
            return False
        return self.names == other.names

    def __hash__(self) -> int:
        return hash(self.names)

    def emit(
        self,
        df: pd.DataFrame,
        synthetic_data_provider: T,
        dependent_columns: list[str] | None = None,
        **kwargs,
    ) -> pd.Series | pd.DataFrame:
        """Invoke the underlying provider to generate column value(s).

        Args:
            df: The partially-built DataFrame whose ``len`` determines the
                number of rows and whose columns may be read for dependencies.
            synthetic_data_provider: The data provider instance (Faker, RNG, …).
            dependent_columns: Names of columns in *df* this attribute depends on.
            **kwargs: Forwarded to the attribute data provider callable.

        Returns:
            A ``pd.Series`` for single-column attributes or a ``pd.DataFrame``
            for multi-column attribute groups.
        """
        return self.attribute_data_provider(
            synthetic_data_provider,
            df,
            dependent_columns,
            **kwargs,
        )


class PaymentDatasetAttribute[T: SyntheticDataProvider](
    PaymentDatasetAttributeGroup[T]
):
    """A single-column attribute — convenience subclass of ``PaymentDatasetAttributeGroup``."""

    def __init__(
        self,
        name: str,
        attribute_data_provider: AttributeDataProviderProtocol[T],
    ):
        super().__init__(
            names=(name,),
            attribute_data_provider=attribute_data_provider,
        )
