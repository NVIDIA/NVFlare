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
