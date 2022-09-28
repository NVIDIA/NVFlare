from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class StatisticsPrivacyCleanser(ABC):
    @abstractmethod
    def apply(self, metrics: dict, client_name: str) -> Tuple[dict, bool]:
        pass

    def cleanse(
        self, metrics: dict, metric_keys: List[str], validation_result: Dict[str, Dict[str, bool]]
    ) -> (dict, bool):
        """
        Args:
            metrics: original client local metrics
            metric_keys: metric keys need to be cleansed
            validation_result: local metrics privacy validation result
        Returns:
            filtered metrics with feature metrics that violating the privacy policy be removed from the original metrics

        """
        metrics_modified = False
        for metric in metric_keys:
            for ds_name in list(metrics[metric].keys()):
                for feature in list(metrics[metric][ds_name].keys()):
                    if not validation_result[ds_name][feature]:
                        metrics[metric][ds_name].pop(feature, None)
                        metrics_modified = True

        return metrics, metrics_modified
