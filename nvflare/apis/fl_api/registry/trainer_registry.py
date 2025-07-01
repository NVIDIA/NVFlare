

_TRAINER_REGISTRY = {}

@property
def get_trainer_registry():
    return _TRAINER_REGISTRY


def register_trainer(name: str):
    def decorator(factory_fn):
        if name not in _TRAINER_REGISTRY:
            _TRAINER_REGISTRY[name] = factory_fn
        else:
            print(f"Trainer '{name}' already registered.")
        return factory_fn
    return decorator


def get_trainer(name: str, **kwargs):
    if name not in _TRAINER_REGISTRY:
        raise ValueError(f"Trainer preset '{name}' not found.")
    return _TRAINER_REGISTRY[name](**kwargs)
