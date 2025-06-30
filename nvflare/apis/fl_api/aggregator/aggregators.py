from typing import List, Any, Tuple


class Aggregator:
    def aggregate(self, updates: List[Any]) -> Any:
        raise NotImplementedError

    def incremental_aggregate(self, global_state: Any, update: Any) -> Any:
        raise NotImplementedError


class FuncAggregatorWrapper(Aggregator):
    def __init__(self, fn):
        self.fn = fn

    def aggregate(self, updates: List[Any]) -> Any:
        return self.fn(updates)

    def incremental_aggregate(self, global_state: Any, update: Any) -> Any:
        # Fallback to re-aggregating if incremental isn't supported
        raise NotImplementedError("incremental_aggregate not supported by function-only aggregator")



class DefaultAggregator:
    def aggregate(self, updates: List[Tuple[Any, int]]) -> Any:
        """
        Aggregate a list of updates using weighted average.
        Each update is a tuple: (value, weight), e.g., (model_params, num_samples)

        Example:
            updates = [(model1, 100), (model2, 200), ...]
        """
        total_weight = sum(weight for _, weight in updates)
        if total_weight == 0:
            raise ValueError("Total weight is zero. Cannot aggregate.")

        # Assume model weights are lists/arrays of floats or tensors
        num_updates = len(updates)
        averaged = None

        for i, (update, weight) in enumerate(updates):
            scale = weight / total_weight
            if averaged is None:
                averaged = self._scale(update, scale)
            else:
                averaged = self._add(averaged, self._scale(update, scale))

        return averaged

    def incremental_aggregate(self, global_state: Any, update: Tuple[Any, int]) -> Any:
        """
        Incrementally update global state using weighted average.

        Args:
            global_state: previous global weights
            update: (new_client_update, weight)
        """
        new_update, weight = update
        return self._average_pair(global_state, new_update, weight)

    def _scale(self, model: Any, factor: float) -> Any:
        # Assume model is a list of floats, numpy arrays, or torch tensors
        return [param * factor for param in model]

    def _add(self, a: Any, b: Any) -> Any:
        return [x + y for x, y in zip(a, b)]

    def _average_pair(self, prev: Any, new: Any, w_new: int) -> Any:
        # Placeholder: assumes equal weight if w_prev is unknown
        # Replace with better logic if you track running weights
        return [(x + y) / 2 for x, y in zip(prev, new)]
