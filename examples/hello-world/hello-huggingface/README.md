# Hello HuggingFace
This example demonstrates how to use NVIDIA FLARE with the HuggingFace Client API
to run federated PEFT/LoRA fine-tuning with a Qwen causal language model. The
complete example code is in the `hello-huggingface` directory. It is recommended
to create a virtual environment and run everything within a virtualenv.

## NVIDIA FLARE Installation
For the complete installation instructions, see
[Installation](https://nvflare.readthedocs.io/en/main/installation.html).

> **Main branch note:** The HuggingFace Client API is introduced for NVFlare
> 2.9.0. Until that package is published, install NVFlare from this repository
> and install the remaining example dependencies separately.

For a released branch:

```
pip install nvflare
```

For the current `main` branch, run these commands from the repository root:

```
python -m pip install -e .
python -m pip install torch transformers accelerate datasets peft trl safetensors
```

The `nvflare~=2.9.0rc` entry in `requirements.txt` records the first compatible
release. After NVFlare 2.9.0 is published,
`python -m pip install -r requirements.txt` installs the complete environment.

## Code Structure
First get the example code from GitHub:

```
git clone https://github.com/NVIDIA/NVFlare.git
```

Then navigate to the `hello-huggingface` directory:

```
git switch <release branch>
cd examples/hello-world/hello-huggingface
```

```bash
hello-huggingface
|
|-- client.py             # HuggingFace/TRL local training script
|-- model.py              # Qwen LoRA server-side model
|-- prepare_data.py       # helper that writes synthetic per-site JSONL data
|-- job.py                # job recipe that defines client and server configurations
|-- requirements.txt      # dependencies
|-- README.md
```

## Data
This example uses small JSONL instruction datasets. The `prepare_data.py` script
writes synthetic per-site data under:

```
/tmp/nvflare/hello-huggingface/data
```

Each row can contain either a single `text` field or the instruction-tuning
fields `instruction`, `input`, and `output`. In a real FL experiment, each site
would store its local files under `<data_root>/<site_name>/`.

`job.py` does not generate or download data. It expects `train.jsonl` and
`valid.jsonl` to exist for each simulated site and reports a clear error if they
are missing.

To use pre-prepared data, set `--data_root` to a directory with this layout:

```bash
data
|
|-- site-1
|   |-- train.jsonl
|   |-- valid.jsonl
|-- site-2
|   |-- train.jsonl
|   |-- valid.jsonl
```

## Model
The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Override
`--model_name_or_path` to use another Qwen causal-LM checkpoint.

The server-side `QwenLoRAModel` is defined in [model.py](model.py). The client
and server exchange only LoRA adapter weights. Full-model and multi-node Qwen
workflows remain in the [advanced LLM example](../../advanced/llm_hf/README.md).

## Client Code
The client code [client.py](client.py) is a standard HuggingFace/TRL
`SFTTrainer` script. The federated adaptation is intentionally small:

```python
import nvflare.client.hf as flare

flare.init()
site_name = flare.get_site_name()

flare.patch(trainer)

while flare.is_running():
    trainer.evaluate()
    trainer.train()
```

`flare.patch(trainer)` wraps the trainer methods so the script can receive the
global model, evaluate it, run the local training budget, and send the result
back to the FL server. Optional settings such as `local_epochs`,
`stream_metrics`, `params_scope`, and `server_key_prefix` are shown as comments
near the `patch()` call in [client.py](client.py), but the default call is enough
for this example.

## Server-Side Workflow
This example uses the
[`FedAvgRecipe`](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html),
which implements the FedAvg workflow:

1. Initialize the global model.
2. For each training round:
   - Send the global model to selected clients.
   - Wait for client updates.
   - Aggregate client models into a new global model.

With the Recipe API, there is no need to write custom server code.

## Job Recipe Code
The `FedAvgRecipe` combines the client training script with the built-in
federated averaging workflow:

```python
recipe = FedAvgRecipe(
    name="hello-huggingface",
    model={
        "class_path": "model.QwenLoRAModel",
        "args": {"model_name_or_path": args.model_name_or_path},
    },
    min_clients=args.n_clients,
    num_rounds=args.num_rounds,
    train_script="client.py",
    train_args=f"--model_name_or_path {args.model_name_or_path} --data_root {data_root}",
    launch_external_process=True,
    server_expected_format=ExchangeFormat.PYTORCH,  # Preserve bf16 tensors.
    key_metric="",  # Disable best-model selection for this API-focused example.
    enable_tensor_disk_offload=True,  # Reduce memory for large tensor payloads.
)
```

The adapter-shaped server model and PEFT trainer expose the same state-dict keys,
so the default `flare.patch(trainer)` call needs no parameter-scope or key-prefix
configuration.

## Prepare Data
Prepare the default two-client synthetic dataset:

```
python prepare_data.py
```

You can also prepare data for a custom number of clients or a custom output
directory:

```
python prepare_data.py --n_clients 4 --data_root /path/to/qwen_jsonl_data
```

The generated data uses this layout:

```bash
qwen_jsonl_data
|
|-- site-1
|   |-- train.jsonl
|   |-- valid.jsonl
|-- site-2
|   |-- train.jsonl
|   |-- valid.jsonl
|-- site-N
|   |-- train.jsonl
|   |-- valid.jsonl
```

## Run Job
After the data is prepared, run:

```
python job.py
```

If you prepared data in a non-default location, pass it explicitly:

```
python job.py --data_root /path/to/qwen_jsonl_data
```

> **Note:** Qwen checkpoints are downloaded from HuggingFace when the example
> runs. The recipe keeps `server_expected_format=ExchangeFormat.PYTORCH` so
> bfloat16 tensors are not converted through NumPy.

## Output Summary
The simulation creates a workspace under:

```
/tmp/nvflare/simulation
```
