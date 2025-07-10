from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field


class DistributedConfig(BaseModel):

    world_size: int = Field(1, description="Total number of processes participating in the job.")
    rank: int = Field(0, description="Unique rank of this process.")
    backend: str = Field("nccl", description="Distributed backend (nccl, gloo, mpi).")
    init_method: str = Field("env://", description="Initialization method for process group.")
    device_ids: Optional[List[int]] = Field(default_factory=lambda: [0], description="GPU device IDs for this process.")
    output_device: Optional[int] = Field(0, description="Device ID for output placement.")
    find_unused_parameters: bool = Field(False, description="Flag to find unused parameters during backward pass.")
    broadcast_buffers: bool = Field(True, description="Flag to broadcast buffers from rank 0.")
    bucket_cap_mb: int = Field(25, description="Bucket size (MB) for gradient communication.")
    timeout_seconds: int = Field(1800, description="Timeout in seconds for distributed operations.")

    accelerator: Optional[str] = Field(
        default=None, description="Device type to use for training. Options: 'cpu', 'gpu', 'tpu'."
    )

    devices: Optional[Union[int, List[int], str]] = Field(
        default=None,
        description=(
            "Number or list of devices to use. "
            "If int: number of devices (e.g., 2 uses device 0 and 1). "
            "If list: specify device indices explicitly (e.g., [0, 2]). "
            "If str: 'auto' lets the framework choose available devices. "
            "Used together with 'accelerator'."
        ),
    )
    strategy: Optional[str] = Field(
        default=None, description="Distributed strategy, e.g. 'ddp', 'ddp_spawn', 'dp', or custom."
    )

    extra: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters.")


class TrainerConfig(BaseModel):
    name: str
    framework: str
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    distributed: Optional[DistributedConfig] = None
    metadata: Optional[Dict[str, Any]] = None
