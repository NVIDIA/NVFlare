from typing import List, Any

from nvflare.apis.fl_api.strategies.base.agg_strategy import AggStrategy


# 8. Federated Statistics (e.g., sum, count, histogram)
class FedStatisticsStrategy(AggStrategy):
    def aggregate(self, updates: List[Any], round_number: int) -> Any:
        total = {}
        for update in updates:
            for k, v in update.items():
                total[k] = total.get(k, 0) + v
        return total
