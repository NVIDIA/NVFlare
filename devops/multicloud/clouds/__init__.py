from .aws import AwsProvider
from .azure import AzureProvider
from .gcp import GcpProvider
from .kubernetes import KubernetesProvider

CLOUD_ORDER = ("gcp", "aws", "azure", "kubernetes")

PROVIDERS = {
    "gcp": GcpProvider(),
    "aws": AwsProvider(),
    "azure": AzureProvider(),
    "kubernetes": KubernetesProvider(),
}


def get_provider(name: str):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown cloud: {name}") from None
