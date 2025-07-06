from typing import List, Any, Dict
import numpy as np
from nvflare.apis.fl_api.message.fl_message import MessageEnvelope, MessageType
from nvflare.apis.fl_api.interfaces.strategy import Strategy

class FedStatsy(Strategy):
    """
    Federated Statistics Aggregator supporting: min, max, mean, std, variance, sum, count, histogram, quantile.
    Two-round protocol:
      - Round 0: stats with no dependencies (sum, count, mean, histogram, min, max, quantile)
      - Round 1: stats depending on global mean (std, variance)
    Aggregation input and output are MessageEnvelope objects.
    """
    def __init__(self, statistic_configs: Dict[str, dict] = None, min_clients: int = 1):
        self.statistic_configs = statistic_configs or {}
        self.min_clients = min_clients
        self.global_mean = None  # For round 1 stats

    def coordinate(self, selected_clients, global_state, round_number, communication, **kwargs):
        # Accept a list of stats to compute in this round
        stats = kwargs.get("stats", ["mean", "sum", "count", "min", "max", "histogram", "quantile"])
        results = {}
        # Inform peers of the target statistics for this round using MessageEnvelope.payload
        msg = MessageEnvelope()
        msg.payload = {"target_stats": stats}
        communication.broadcast_and_wait(selected_clients, msg)
        if round_number == 0:
            updates = communication.receive_from_peers(selected_clients)
            for stat in stats:
                result_msg = self.aggregate(updates, stat, round_number=0, **kwargs)
                results[stat] = result_msg
                if stat == "mean":
                    self.global_mean = result_msg.model_state["mean"]
            # Optionally broadcast the global mean if it will be needed for round 1
            next_stats = kwargs.get("next_stats", [])
            if any(s in next_stats for s in ("variance", "std")):
                mean_msg = MessageEnvelope()
                mean_msg.payload = {"global_mean": self.global_mean}
                communication.broadcast_and_wait(selected_clients, mean_msg)
            return results
        elif round_number == 1:
            # Instruct clients to use the global mean (already sent) for their local computation
            updates = communication.receive_from_peers(selected_clients)
            for stat in stats:
                results[stat] = self.aggregate(updates, stat, round_number=1, **kwargs)
            return results
        else:
            raise ValueError(f"Unsupported round_number: {round_number}")

    def aggregate(self, updates: List[MessageEnvelope], stat: str, round_number: int = 0, **kwargs) -> MessageEnvelope:
        # Type check: all updates must be MessageEnvelope
        for u in updates:
            if not isinstance(u, MessageEnvelope):
                raise TypeError(f"All updates must be MessageEnvelope, got {type(u)}")
        # Round 0: aggregate basic stats
        if round_number == 0:
            values = [u.model_state[stat] for u in updates if hasattr(u, "model_state") and stat in u.model_state]
            if not values:
                raise ValueError(f"No values found for stat '{stat}' in updates.")
            if stat == "min":
                result = np.min(values, axis=0)
            elif stat == "max":
                result = np.max(values, axis=0)
            elif stat == "mean":
                result = np.mean(values, axis=0)
                self.global_mean = result
            elif stat == "sum":
                result = np.sum(values, axis=0)
            elif stat == "count":
                result = np.sum([u.model_state.get("count", 1) for u in updates if hasattr(u, "model_state")])
            elif stat == "histogram":
                hists, bin_edges = zip(*values)
                total_hist = np.sum(hists, axis=0)
                result = (total_hist, bin_edges[0])
            elif stat == "quantile":
                q = kwargs.get("q", 0.5)
                result = np.quantile(np.stack(values, axis=0), q, axis=0)
            else:
                raise ValueError(f"Unsupported stat for round 0: {stat}")
        # Round 1: stats that depend on global mean
        elif round_number == 1:
            if self.global_mean is None:
                raise RuntimeError("Global mean must be computed in round 0 before std/variance aggregation.")
            # Each update should provide sum((x_i - mean)^2) and count
            if stat in ("variance", "std"):
                sq_diffs = [u.model_state["sq_diff"] for u in updates if hasattr(u, "model_state") and "sq_diff" in u.model_state]
                counts = [u.model_state.get("count", 1) for u in updates if hasattr(u, "model_state")]
                total_sq_diff = np.sum(sq_diffs, axis=0)
                total_count = np.sum(counts)
                variance = total_sq_diff / max(total_count, 1)
                if stat == "variance":
                    result = variance
                else:
                    result = np.sqrt(variance)
            else:
                raise ValueError(f"Unsupported stat for round 1: {stat}")
        else:
            raise ValueError(f"Unsupported round_number: {round_number}")
        output = MessageEnvelope()
        output.model_state = {stat: result}
        return output
