from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from nvflare.app_common.app_constant import StatisticsConstants as StC


class StatisticsPrivacyCleanser(ABC):
    @abstractmethod
    def apply(self, statistics: dict, client_name: str) -> Tuple[dict, bool]:
        pass

    def cleanse(
        self, statistics: dict, statistic_keys: List[str], validation_result: Dict[str, Dict[str, bool]]
    ) -> (dict, bool):
        """
        Args:
            statistics: original client local metrics
            statistic_keys: statistic keys need to be cleansed
            validation_result: local metrics privacy validation result
        Returns:
            filtered metrics with feature metrics that violating the privacy policy be removed from the original metrics

        """
        statistics_modified = False
        for key in statistic_keys:
            if key != StC.STATS_COUNT:
                for ds_name in list(statistics[key].keys()):
                    for feature in list(statistics[key][ds_name].keys()):
                        if not validation_result[ds_name][feature]:
                            statistics[key][ds_name].pop(feature, None)
                            statistics_modified = True

        return statistics, statistics_modified
