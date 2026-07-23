# Qwen HuggingFace Client API

This example shows the `nvflare.client.hf` API with a Qwen causal language model.
It keeps the example layout intentionally small:

```text
hf_client_api/
|-- client.py
|-- job.py
|-- model.py
|-- requirements.txt
`-- README.md
```

`client.py` is a regular HuggingFace/TRL `SFTTrainer` script. The federated
change is the import of `nvflare.client.hf`, one `flare.patch(trainer)` call,
and a normal `while flare.is_running(): trainer.evaluate(); trainer.train()`
round loop.

By default, `job.py` prepares tiny synthetic JSONL files under
`/tmp/nvflare/hf_client_api_qwen/data` and runs LoRA fine-tuning with
`Qwen/Qwen2.5-0.5B-Instruct`. Override `--model_name_or_path` to use another
Qwen causal-LM checkpoint.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` uses `nvflare~=2.9.0rc1`, the first upcoming NVFlare release
with `nvflare.client.hf`. Until that package is published, install NVFlare from
this repository before running the example.

Run a two-client simulation:

```bash
python job.py
```

Export the job without running it:

```bash
python job.py --export_config --job_dir /tmp/nvflare/jobs/qwen_hf_client_api
```

Run full-model SFT instead of LoRA:

```bash
python job.py --train_mode sft
```

For large Qwen models, use `--train_mode peft` and keep
`server_expected_format=ExchangeFormat.PYTORCH` so bfloat16 tensors are not
converted through NumPy.
