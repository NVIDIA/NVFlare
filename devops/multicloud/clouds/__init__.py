from .aws import AwsProvider
from .azure import AzureProvider
from .gcp import GcpProvider

CLOUD_ORDER = ("gcp", "aws", "azure")

PROVIDERS = {
    "gcp": GcpProvider(),
    "aws": AwsProvider(),
    "azure": AzureProvider(),
}


def get_provider(name: str):
    try:
        return PROVIDERS[name]
    except KeyError:
        raise ValueError(f"unknown cloud: {name}") from None
