# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from tqdm import tqdm

from .evaluation.evaluate_utils import PerformanceMeter
from .utils import get_output, to_cuda


def local_train(
    idx,
    cr,
    local_epochs,
    tasks,
    train_dl,
    model,
    optimizer,
    scheduler,
    criterion,
    scaler,
    train_loss,
    local_rank,
    fp16,
    writer=None,
    **args,
):
    """
    Train local_epochs on the client model
    """

    model.train()

    for epoch in range(local_epochs):
        # Set epoch for sampler if it exists and has set_epoch method
        if hasattr(train_dl, 'sampler') and hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(cr * local_epochs + epoch)

        for batch in tqdm(
            train_dl,
            desc="CR %d Local Epoch %d Net %d Task: %s" % (cr, epoch + 1, idx + 1, ",".join(tasks)),
            disable=(local_rank != 0),
        ):
            optimizer.zero_grad()
            batch = to_cuda(batch)
            images = batch['image']

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                outputs = model(images)
                loss_dict = criterion(outputs, batch, tasks)

            # Log loss values
            for task in tasks:
                loss_value = loss_dict[task].detach().item()
                batch_size = outputs[task].size(0)
                train_loss[task].update(loss_value / batch_size, batch_size)

            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()

        # Log epoch losses to TensorBoard
        if writer is not None:
            for task in tasks:
                writer.add_scalar(f"train_loss_epoch/{task}", train_loss[task].avg, cr * local_epochs + epoch)

        scheduler.step(cr * local_epochs + epoch)


def eval_metric(tasks, dataname, val_dl, model, idx, **args):
    """
    Evaluate client model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating Net %d Task: %s" % (idx, ",".join(tasks))):
            batch = to_cuda(batch)
            images = batch['image']
            # Handle both wrapped and unwrapped models
            if hasattr(model, 'module'):
                outputs = model.module(images)
            else:
                outputs = model(images)
            performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

    eval_results = performance_meter.get_score()

    results_dict = {}
    for task in tasks:
        for key in eval_results[task]:
            results_dict['eval/' + str(idx) + '_' + task + '_' + key] = eval_results[task][key]

    return results_dict
