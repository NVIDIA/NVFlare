
from typing import Callable, Dict
from nvflare.apis.fl_api.strategies.base.strategy import Strategy

_STRATEGY_REGISTRY: Dict[str, Callable[..., "Strategy"]] = {}


def register_strategy(name: str):
    def decorator(cls_or_fn):
        if name not in _STRATEGY_REGISTRY:
            _STRATEGY_REGISTRY[name.lower()] = cls_or_fn
        else:
            print(f"Strategy '{name}' already registered.")
        return cls_or_fn
    return decorator


def get_strategy(name: str, **kwargs) -> "Strategy":
    if name.lower() not in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found in registry.")
    return _STRATEGY_REGISTRY[name.lower()](**kwargs)
