# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import logging
import random
import re
import time
from datetime import datetime

import torch

# Client API
import nvflare.client as flare

logger = logging.getLogger("POCExecutor")


def main():

    flare.init()
    input_model = flare.receive()

    print("@@@ input_model: ", input_model)
    round_num = input_model.current_round
    print("@@@ Round number in this round: ", round_num)

    # 'site-2'
    site_name = input_model.meta.get("site_name")
    multiplier = re.search(r"\d+", site_name).group()
    print("@@@ site_name: ", site_name)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # data shape
    if input_model.current_round == 0:
        # Not involving init random weights
        weight = torch.zeros([5, 10], dtype=torch.float32)
        bias = torch.zeros([5], dtype=torch.float32)
    else:
        weight = input_model.params.get("fc1.weight")
        bias = input_model.params.get("fc1.bias")

    # do the job
    zzz = random.uniform(1.0, 3.0)
    print("@@@ Sleep " + str(zzz))
    time.sleep(zzz)

    weight = torch.add(weight, 1) * int(multiplier)
    bias = torch.add(bias, 1) * int(multiplier)

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("Finished Training")

    params = {
        "fc1.weight": weight,
        "fc1.bias": bias,
    }

    def evaluate(data):
        # between 0-1, increase with time
        t = time.time() / 1e10
        print(f"fake evaluate data: {data}")
        print(
            f"fake evaluate result: {t}, generated at {datetime.utcfromtimestamp(t * 1e10).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return t

    accuracy = evaluate(params)

    output_model = flare.FLModel(
        params=params,
        metrics={"accuracy": accuracy},
        meta={"NUM_STEPS_CURRENT_ROUND": 2, "start": start_time, "end": end_time},
    )

    print("@@@ output_model: ", output_model)
    flare.send(output_model)


if __name__ == "__main__":
    main()
