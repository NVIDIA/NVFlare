from typing import Any


def wrap_with_dict(data: Any):
    return {"any": data}


def unwrap_dict(d: dict) -> Any:
    return d.get("any")
