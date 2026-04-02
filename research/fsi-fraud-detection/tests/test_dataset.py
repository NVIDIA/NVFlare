"""Tests for the dataset generation pipeline (topological_sort + generate)."""

import pandas as pd
import pytest
from data_generation.dataset import generate, topological_sort
from data_generation.dataset_attribute import PaymentDatasetAttribute, PaymentDatasetAttributeGroup
from data_generation.synthetic_data_provider.synthetic_data_provider import SyntheticDataProvider

# ---------------------------------------------------------------------------
# Minimal provider and generator stubs
# ---------------------------------------------------------------------------


class StubProvider(SyntheticDataProvider):
    """A no-op provider for testing."""

    def provide(self):
        return None


def _make_attr(name: str) -> PaymentDatasetAttributeGroup:
    """Create a single-column attribute that fills with the column name."""

    def _gen(provider, df, dependent_columns=None, **kwargs):
        return pd.Series([name] * len(df), index=df.index)

    return PaymentDatasetAttribute(name, _gen)


def _make_group(names: tuple[str, ...]):
    """Create a multi-column attribute group filling each column with its name."""

    def _gen(provider, df, dependent_columns=None, **kwargs):
        return pd.DataFrame(
            {f"col_{i}": [n] * len(df) for i, n in enumerate(names)},
            index=df.index,
        )

    return PaymentDatasetAttributeGroup(names, _gen)


def _make_dependent_attr(name: str) -> PaymentDatasetAttributeGroup:
    """Create an attribute that concatenates its dependency values."""

    def _gen(provider, df, dependent_columns=None, **kwargs):
        cols = dependent_columns or []
        combined = df[cols[0]].astype(str)
        for c in cols[1:]:
            combined = combined + "+" + df[c].astype(str)
        return pd.Series(combined, index=df.index)

    return PaymentDatasetAttribute(name, _gen)


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    def test_empty_graph(self):
        assert topological_sort({}) == []

    def test_single_independent(self):
        a = _make_attr("A")
        result = topological_sort({a: []})
        assert result == [a]

    def test_linear_chain(self):
        a = _make_attr("A")
        b = _make_attr("B")
        c = _make_attr("C")
        graph = {a: [], b: ["A"], c: ["B"]}
        result = topological_sort(graph)
        names = [attr.names for attr in result]
        assert names.index(("A",)) < names.index(("B",))
        assert names.index(("B",)) < names.index(("C",))

    def test_diamond_dependency(self):
        """A → B, A → C, B+C → D."""
        a = _make_attr("A")
        b = _make_attr("B")
        c = _make_attr("C")
        d = _make_attr("D")
        graph = {a: [], b: ["A"], c: ["A"], d: ["B", "C"]}
        result = topological_sort(graph)
        names = [attr.names for attr in result]
        assert names.index(("A",)) < names.index(("B",))
        assert names.index(("A",)) < names.index(("C",))
        assert names.index(("B",)) < names.index(("D",))
        assert names.index(("C",)) < names.index(("D",))

    def test_multi_column_group(self):
        g = _make_group(("X", "Y"))
        dep = _make_attr("Z")
        graph = {g: [], dep: ["X"]}
        result = topological_sort(graph)
        assert result.index(g) < result.index(dep)

    def test_cycle_raises(self):
        a = _make_attr("A")
        b = _make_attr("B")
        graph = {a: ["B"], b: ["A"]}
        with pytest.raises(ValueError, match="[Cc]ycle"):
            topological_sort(graph)

    def test_preserves_all_attributes(self):
        attrs = [_make_attr(c) for c in "ABCDE"]
        graph = {a: [] for a in attrs}
        result = topological_sort(graph)
        assert set(result) == set(attrs)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_basic_generation(self):
        a = _make_attr("A")
        b = _make_attr("B")
        graph = {a: [], b: []}
        providers = {a: StubProvider(), b: StubProvider()}
        df = generate(graph, providers, n_rows=5)
        assert len(df) == 5
        assert list(df.columns) == ["A", "B"]
        assert (df["A"] == "A").all()
        assert (df["B"] == "B").all()

    def test_dependent_generation(self):
        a = _make_attr("A")
        b = _make_dependent_attr("B")
        graph = {a: [], b: ["A"]}
        providers = {a: StubProvider(), b: StubProvider()}
        df = generate(graph, providers, n_rows=3)
        # B should contain "A" since it concatenates dep column values
        assert (df["B"] == "A").all()

    def test_multi_column_group(self):
        g = _make_group(("X", "Y"))
        graph = {g: []}
        providers = {g: StubProvider()}
        df = generate(graph, providers, n_rows=4)
        assert "X" in df.columns
        assert "Y" in df.columns
        assert (df["X"] == "X").all()
        assert (df["Y"] == "Y").all()

    def test_missing_provider_raises(self):
        a = _make_attr("A")
        graph = {a: []}
        with pytest.raises(KeyError):
            generate(graph, {}, n_rows=1)

    def test_kwargs_forwarded(self):
        """Ensure **kwargs from generate() reach the attribute generator."""
        received = {}

        def _capture(provider, df, dependent_columns=None, **kwargs):
            received.update(kwargs)
            return pd.Series([0] * len(df), index=df.index)

        a: PaymentDatasetAttributeGroup = PaymentDatasetAttribute("A", _capture)
        graph: dict[PaymentDatasetAttributeGroup, list[str]] = {a: []}
        providers: dict[PaymentDatasetAttributeGroup, SyntheticDataProvider] = {a: StubProvider()}
        generate(graph, providers, n_rows=1, my_param="hello")
        assert received["my_param"] == "hello"

    def test_correct_row_count(self):
        a = _make_attr("A")
        graph = {a: []}
        providers = {a: StubProvider()}
        for n in [0, 1, 100]:
            df = generate(graph, providers, n_rows=n)
            assert len(df) == n
