# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist

import nvflare.client as flare


def _broadcast_object_from_rank0(obj):
    objects = [obj if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def main():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"expected two torchrun ranks, got world_size={world_size}")

    flare.init(rank=rank)

    received_model = flare.receive()
    if rank == 0:
        if received_model is None:
            raise RuntimeError("rank 0 expected an FLModel from NVFlare")
    elif received_model is not None:
        raise RuntimeError("nonzero rank should not receive directly from NVFlare")

    input_model = _broadcast_object_from_rank0(received_model)
    if input_model is None or input_model.params is None:
        raise RuntimeError("distributed broadcast did not provide the FLModel to all ranks")

    rank_contribution = torch.tensor(float(rank + 1))
    dist.all_reduce(rank_contribution, op=dist.ReduceOp.SUM)
    if rank_contribution.item() != 3.0:
        raise RuntimeError(f"expected rank contribution sum 3.0, got {rank_contribution.item()}")

    params = {name: tensor.detach().clone() + rank_contribution.item() for name, tensor in input_model.params.items()}
    output_model = flare.FLModel(
        params=params,
        metrics={
            "accuracy": rank_contribution.item(),
            "torchrun_world_size": world_size,
        },
        meta={
            "NUM_STEPS_CURRENT_ROUND": world_size,
            "torchrun_rank": rank,
        },
    )

    flare.send(output_model)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
