# Hello PyTorch

This quickstart trains a small image classifier with federated averaging (FedAvg). Two simulated clients train on
independently generated local datasets for three rounds, and then evaluate the persisted final global model on
separate evaluation data. The zero-argument path is deterministic, runs on CPU, downloads no dataset, and requires no
tracking service.

## Install

Create and activate a virtual environment, then get the source and enter the
example directory:

```bash
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare/examples/hello-world/hello-pt
```

Install the example dependencies from that directory:

```bash
python -m pip install -r requirements.txt
```

For other installation options, see the [NVFlare installation guide](https://nvflare.readthedocs.io/en/main/installation.html).

## Run the default quickstart

```bash
python job.py
```

The default run uses:

| Setting | Default |
| --- | --- |
| Dataset | Deterministic synthetic images with a class-related signal |
| Clients | 2 |
| Federated rounds | 3 |
| Local epochs per round | 1 |
| Training examples per client | 200 |
| Evaluation examples per client | 100 |
| Batch size | 32 |
| Data-loader workers | 0 |
| Experiment tracking | Off |
| Post-training evaluation | Final global model on both clients |

Results are written under `/tmp/nvflare/simulation/hello-pt`. The primary result artifacts are:

- `server/simulate_job/app_server/FL_global_model.pt`: the persisted final global model.
- `server/simulate_job/metrics/metrics_summary.json`: final aggregated training-round metrics and available best-model
  metric metadata.
- `server/simulate_job/cross_site_val/cross_val_results.json`: post-training evaluation of the persisted final model by
  client.

Use `metrics_summary.json` for a compact summary of the federated training metrics. To inspect the accuracy of the
persisted model after the last aggregation, use the `SRV_FL_global_model.pt` result for each site in
`cross_val_results.json`. A `best` model can also appear because training-round metrics score the global model received
at the start of a round; the final aggregate is produced after the last such score. The final model can therefore
outperform both the last reported training-round accuracy and the model selected earlier as `best` in this short run.
The automated acceptance test requires
at least 60% accuracy on both sites and at least a 40 percentage-point improvement over the initial global model.
These thresholds are calibrated to the fixed model and data seeds with the three-round default. They verify that this
specific federated run changed the model meaningfully; they are not guarantees for other initializations or
hyperparameters and are not benchmark claims.

## Code structure

```text
hello-pt/
├── client.py          # Client-side training and evaluation
├── job.py             # FedAvg recipe and simulation entry point
├── model.py           # PyTorch model definition and deterministic initialization
├── prepare_data.py    # Default data generation and optional CIFAR-10 download
├── requirements.txt   # Default dependencies
└── README.md
```

## Client-side workflow

Most of [`client.py`](client.py) is ordinary PyTorch training code. The block below is the actual task loop from that
file, with only its indentation normalized. The complete `main()` initializes `client_name`, `model`, `optimizer`,
`loss`, `train_loader`, `test_loader`, `device`, `summary_writer`, `last_params`, and the parsed `args` immediately
before this loop. The module imports `torch` and `nvflare.client` (as `flare`), defines `LOCAL_MODEL_PATH`, and calls
`flare.init()` before entering the loop.

```python
while flare.is_running():
    # (4) receives FLModel from NVFlare
    input_model = flare.receive()
    print(f"site = {client_name}, current_round={input_model.current_round}")

    # Cross-site evaluation requests the client's latest local model without
    # sending model parameters in the request.
    if flare.is_submit_model():
        if last_params is None:
            error_msg = "submit_model called before a local model was trained"
            print(f"ERROR: {error_msg}")
            # TaskScriptRunner converts this exception into TOPIC_ABORT so the
            # executor can report the task failure instead of waiting for a result.
            raise RuntimeError(error_msg)
        print(f"site = {client_name}, submitting local model")
        flare.send(flare.FLModel(params=last_params))
        continue

    # (5) loads model from NVFlare
    model.load_state_dict(input_model.params)
    # (6) evaluate the received global model before local training
    accuracy_before_training = evaluate(model, test_loader, device)

    # (optional) Task branch for cross-site evaluation
    if flare.is_evaluate():
        print(f"site = {client_name}, running cross-site evaluation")
        # For CSE, just return the evaluation metrics without training
        output_model = flare.FLModel(metrics={"accuracy": accuracy_before_training})
        flare.send(output_model)
        continue

    model.train()
    steps = args.epochs * len(train_loader)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            predictions = model(images)
            cost = loss(predictions, labels)
            cost.backward()
            optimizer.step()

            running_loss += cost.item()
        avg_loss = running_loss / len(train_loader)
        print(f"site={client_name}, epoch={epoch + 1}/{args.epochs}, loss={avg_loss:.4f}")
        global_step = input_model.current_round * args.epochs + epoch
        summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=global_step)

    print(f"Finished Training for {client_name}")
    trained_accuracy = evaluate(model, test_loader, device)

    last_params = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
    torch.save(last_params, LOCAL_MODEL_PATH)

    # (7) construct trained FL model
    output_model = flare.FLModel(
        params=last_params,
        # The primary metric evaluates the received global model, which is
        # the model the server considers for best-model selection. Report
        # the trained local model separately to make progress visible.
        metrics={
            "accuracy": accuracy_before_training,
            "accuracy_after_local_training": trained_accuracy,
        },
        meta={"NUM_STEPS_CURRENT_ROUND": steps},
    )
    print(f"site: {client_name}, sending model to server.")
    # (8) send model back to NVFlare
    flare.send(output_model)
```

The linked file is the runnable source; this excerpt introduces no alternate helper functions or signatures.

## Server-side workflow

[`job.py`](job.py) uses `FedAvgRecipe`, so the example does not need custom server code:

```python
recipe = FedAvgRecipe(
    name="hello-pt",
    min_clients=2,
    num_rounds=3,
    model=create_model(),
    train_script="client.py",
    train_args="...",
)
add_final_global_evaluation(recipe)
recipe.execute(SimEnv(num_clients=2))
```

The recipe initializes the global model, sends it to selected clients, collects local updates, performs weighted
FedAvg aggregation, persists the result, and requests the final evaluation.

Training data remains local to each client. Clients send model parameters, evaluation metrics, and the number of
completed optimizer steps; they do not send their training examples to the server.

## Customize the run

See all options:

```bash
python job.py --help
```

Use CIFAR-10 instead of the deterministic quickstart data:

Simulated clients share the default cache at `/tmp/nvflare/data`. Download both
splits once before starting the simulation; clients open the prepared data
without downloading so they cannot race while writing the same files:

```bash
python prepare_data.py
```

```bash
python job.py --dataset cifar10
```

To use another cache location, pass the same client-local path to both commands:

```bash
python prepare_data.py --data_root /data/cifar
python job.py --dataset cifar10 --data_root /data/cifar
```

All simulated clients then read the same logical CIFAR-10 training and test
datasets from that shared cache. This option is useful for experimentation but
does not demonstrate a federated data partition. The default synthetic path
remains offline and independently generates every site's training and
evaluation samples from the same IID distribution.

The beginner entry point intentionally exposes only the number of clients,
number of rounds, dataset choice, and its client-local data root. Environment
selection, experiment tracking, full cross-site evaluation, external-process
execution, and memory tuning belong in the environment-continuity follow-up
rather than the first federated-learning run.

## Export a deployable job

```bash
python job.py --export --export-dir /tmp/nvflare/jobs/job_config
```

The exported job is written under `/tmp/nvflare/jobs/job_config/hello-pt`.

## Notebook

For an interactive CIFAR-10 and TensorBoard-oriented variant, see [`hello-pt.ipynb`](hello-pt.ipynb). The canonical
deterministic quickstart and its tested defaults are defined by `job.py`.

## Continue to POC and Production

After completing the simulation, continue with the
[advanced environment-continuity example](../../advanced/hello-pt-environments/README.md)
to run the same learning application in a local POC or an already-running production deployment.
