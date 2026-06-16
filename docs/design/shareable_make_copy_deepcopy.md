# Shareable Deep Copy With No-Copy Values

## Objective

Make `make_copy()` produce an isolated `Shareable` copy without duplicating large binary, array, or caller-declared payload objects.

The copied `Shareable` should protect normal nested containers and headers from accidental mutation sharing while preserving identity for values that are expensive or undesirable to copy, such as NumPy arrays and framework tensors.

## Background

`Shareable` is the dictionary-like payload container used for server-client communication. It also stores FLARE headers under the reserved header key.

Before this design, `make_copy()` used `copy.copy(source)`. That behavior had two important properties:

- payload contents were shallow-copied and nested containers remained shared with the source
- headers were deep-copied into a new header dictionary

The shallow payload copy avoided duplicating large objects, but it also meant changes to nested payload containers in the returned copy could mutate the source. A plain `copy.deepcopy(source)` would fix container isolation, but it would also copy large mutable values such as `bytearray` and `numpy.ndarray`, increasing memory use and changing identity-sensitive behavior.

## Goals

- Deep-copy `Shareable` payload containers and headers.
- Preserve identity for built-in no-copy value types: `bytes`, `bytearray`, `memoryview`, and `numpy.ndarray` when NumPy is installed.
- Allow callers to extend the no-copy set with `no_copy_types`.
- Support `no_copy_types` as a single type, tuple of types, or list of types.
- Find no-copy values anywhere in the object graph, including dictionaries, sequences, object attributes, and slotted classes.
- Keep `exclude_headers` behavior scoped to the returned copy.
- Keep NumPy optional.

## Non-Goals

- Do not make mutable no-copy values immutable or copy-on-write.
- Do not add framework-specific imports such as PyTorch.
- Do not replace Python's `copy.deepcopy()` with a custom deep-copy implementation.
- Do not add a global registry for application-specific no-copy types.
- Do not change `Shareable` header storage semantics.

## API

```python
def make_copy(source: Shareable, exclude_headers: list = None, no_copy_types: _NoCopyTypes = None) -> Shareable:
    ...
```

`no_copy_types` accepts:

- `None`
- a single type
- `tuple[type, ...]`
- `list[type]`

The built-in no-copy types are always included. Caller-supplied types extend that base set.

Example:

```python
import torch

copied = make_copy(source, no_copy_types=(torch.Tensor,))
```

## Design

The implementation keeps Python's normal `deepcopy()` behavior, but pre-seeds the `deepcopy` memo with values that must be reused by identity.

```text
make_copy()
  |
  +--> _make_no_copy_memo(source, no_copy_types)
  |      |
  |      +--> _normalize_no_copy_types(no_copy_types)
  |      |
  |      +--> _collect_no_copy_values(source, normalized_types, memo, seen)
  |
  +--> copy.deepcopy(source, memo=memo)
  |
  +--> remove excluded headers from the copied headers
  |
  +--> return copied Shareable
```

`copy.deepcopy()` treats a populated `memo` entry as an already-copied object. By setting `memo[id(value)] = value` for every discovered no-copy value, the returned `Shareable` reuses those values by identity while deep-copying all other containers and objects normally.

### No-Copy Type Normalization

`_normalize_no_copy_types()` combines the built-in no-copy value types with caller-supplied types.

The default built-in set is:

```python
(bytes, bytearray, memoryview)
```

If NumPy imports successfully, `np.ndarray` is added. If NumPy is unavailable, `make_copy()` still works and only the Python built-in no-copy types are used.

### Object Graph Pre-Scan

`_collect_no_copy_values()` traverses the source graph before `deepcopy()` runs. It records all values whose type matches the normalized no-copy type tuple.

Traversal covers:

- dictionary keys and values
- `list`, `tuple`, `set`, and `frozenset` items
- attributes stored in `__dict__`
- attributes declared with `__slots__`

The scan uses a `seen` set of object IDs to avoid infinite recursion on cyclic object graphs.

### Slotted Object Support

Some wrapper types use `__slots__` and do not expose `__dict__`. Without slot traversal, a custom no-copy value such as a tensor stored in a slotted wrapper would not be discovered before `deepcopy()`.

`_iter_slot_names()` walks the class MRO, normalizes string and dictionary slot declarations, skips `__dict__` and `__weakref__`, and handles private name-mangled slot names.

This lets the pre-scan find no-copy values in wrappers such as:

```python
class SlottedWrapper:
    __slots__ = ("nested", "tensor")
```

### Header Handling

Headers are deep-copied as part of the full `Shareable` deep copy. After the copy is created, `exclude_headers` keys are removed from the copied header dictionary only.

The source `Shareable` headers are not modified.

## Behavioral Contract

- The returned object is a new `Shareable` instance.
- Normal nested containers in payloads and headers are deep-copied.
- Built-in and caller-supplied no-copy values are reused by identity.
- Mutable no-copy values are intentionally shared; mutations to those values are visible from both source and copy.
- A no-copy value nested inside a copied object remains shared, while the wrapper object itself may still be deep-copied.
- Header exclusions apply only to the returned copy.

## Edge Cases

- Cyclic graphs are safe during the pre-scan because visited object IDs are tracked.
- No-copy values used as dictionary keys are found as well as no-copy values stored in dictionary values.
- Objects with both `__slots__` and `__dict__` are scanned through both paths.
- Private slotted attributes are found through Python's name-mangling convention.
- When NumPy is not installed, `numpy.ndarray` is not part of the built-in no-copy set.

## Alternatives Considered

### Keep The Shallow Copy

Keeping `copy.copy(source)` would preserve large object identity, but nested payload containers would still be shared. That fails the isolation goal.

### Use Plain Deep Copy

Plain `copy.deepcopy(source)` gives strong isolation, but it can duplicate large arrays and mutable byte buffers. This increases peak memory and breaks identity-preserving expectations for payload values that should be reused.

### Hand-Roll A Deep Copy

A custom recursive copy routine would make no-copy decisions directly, but it would need to duplicate much of Python's `deepcopy()` behavior for custom objects, cycles, and container types. Pre-seeding the `deepcopy` memo is smaller and relies on standard library semantics.

### Only Check Top-Level Payload Values

Top-level checks would miss tensors or arrays stored inside nested containers or wrapper objects. The pre-scan must walk the graph to make the `no_copy_types` contract reliable.

## Risks And Tradeoffs

- Pre-scanning adds graph traversal work before `deepcopy()`. This overhead is proportional to the source graph size, but it avoids copying large values and keeps standard `deepcopy()` semantics.
- Mutable no-copy values are shared by design. The docstring calls this out so callers know that identity reuse also means mutation visibility.
- Behavior differs slightly depending on whether NumPy is importable. This is intentional so the core API remains usable without requiring NumPy.
- If callers pass invalid `no_copy_types`, tuple normalization or `isinstance()` will raise according to normal Python type-checking behavior.

## Design-Relevant Files

- `nvflare/apis/shareable.py`
- `tests/unit_test/apis/shareable_test.py`

## Test Coverage

Unit coverage verifies that:

- `make_copy()` returns a distinct `Shareable`
- nested payload containers are deep-copied
- headers are deep-copied
- excluded headers are removed only from the copy
- built-in no-copy values are reused by identity
- custom no-copy types are reused by identity
- custom no-copy types work when passed as a single type
- custom no-copy values inside slotted wrappers are discovered and reused
- copied wrapper objects still isolate their ordinary nested containers
