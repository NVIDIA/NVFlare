from pydantic import BaseModel
from typing import Optional, Dict, Any


class TrainerConfig(BaseModel):
    name: str  # Optional identifier (e.g. "site_A")
    framework: str  # e.g. "torch", "lightning", "xgboost"
    hyperparams: Dict[str, Any]  # e.g. {"lr": 0.01, "batch_size": 32}
    metadata: Optional[Dict[str, Any]] = None  # Any additional info


