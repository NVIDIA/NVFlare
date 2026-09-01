# FedCoRe: Federated Cross-Modal Representation Completion

This is a starter example for FedCoRe, a federated cross-modal representation completion method. Clients with valid
paired supervision learn a missing-modality completion operator, which is then shared with clients that do not observe
the modality. The project provides a small, public, end-to-end NVIDIA FLARE workflow using Qwen3-VL and MNIST
image-plus-text data.

The accompanying paper is available on arXiv: [FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning](https://arxiv.org/abs/2608.18311).

![Complete-modality evaluation assumptions contrasted with uneven modality availability across federated clients.](figures/fedcore_modality_gap.png)

## Objective

Demonstrate federated cross-modal completion with public model weights and a familiar dataset. Clients that observe
digit images learn an additive classifier-logit correction, clients without paired image supervision have zero
aggregation weight, and validation can retain the identity operator when completion is unsupported.

## Background

Multimodal FL clients often differ in which inputs they observe. A client without a modality cannot directly train a
completion operator for that modality, but it can consume an operator learned from other clients with paired
modality-present and modality-removed passes. FedCoRe formalizes that valid-supervision exchange without reconstructing
raw images.

## MNIST data and client design

Each example contains an MNIST handwritten-digit image and a secondary text report from a simulated noisy OCR sensor.
Digits `0` through `4` are class `A`; digits `5` through `9` are class `B`. The image is authoritative. The OCR report
contains an estimated digit and a `high` or `low` confidence value.

In the default `recoverable` scenario, the OCR estimate has the requested class-level accuracy and high confidence is
more common when the estimate is correct. In the `uninformative` control, OCR class and confidence are exactly balanced
independently of the true class within every client split. Modality availability differs across three disjoint
simulated clients:

![Visual example of the MNIST task and image availability across the three federated clients.](figures/fedcore_synthetic_client_design.svg)

The top row shows one paired training example: the same OCR context is processed with and without its digit image, and
their logit difference supervises the completion operator. The bottom row shows which clients can contribute paired
updates and which clients only use the aggregated operator at inference.

| Client | Training image availability | Role |
| --- | ---: | --- |
| `site-1` | 100% | Paired image-present/image-removed supervision |
| `site-2` | 50% | Example-mixed supervision and deployment examples |
| `site-3` | 0% | Missing-image deployment only; sends no completion update |

MNIST source indices are allocated without replacement across all sites and splits. Training and validation examples
come from disjoint portions of the official training set; test examples come from the official test set. This task
verifies the workflow and is not evidence of clinical effectiveness.

```mermaid
flowchart LR
    A[Prepare local MNIST image + OCR records] --> B[Qwen3-VL image-present and image-removed passes]
    B --> C[Train local logit-effect completion on paired records]
    C --> D[Valid-supervision FedAvg]
    D --> E[Validation-constrained alpha selection]
    E --> F[Evaluate image-missing clients]
```

For an image-present record, the frozen predictor produces `full_logit`; removing the image produces
`missing_logit` and `missing_hidden`. The exchanged completion model predicts an additive residual:

```text
completed_logit = missing_logit + alpha * completion(missing_hidden)
```

Only paired records can supervise the image contribution `full_logit - missing_logit`. A client with no paired
records sends empty model parameters, which makes the aggregator skip its update; it also reports
`NUM_STEPS_CURRENT_ROUND=0` to make the lack of valid supervision explicit. Validation selection combines only
per-client loss sums and counts. The optional pooled test AUROC is a single-machine tutorial diagnostic, not a
production federated metric.

## Repository layout

| Path | Purpose |
| --- | --- |
| `prepare_data.py` | Download MNIST and create deterministic, disjoint client data plus Qwen SFT JSON. |
| `cache_features.py` | Run Qwen3-VL image-present/image-removed passes. |
| `model.py` | Define the lightweight classifier-logit completion operator. |
| `client.py` | Train on valid paired records and implement empty-update behavior. |
| `job.py` | Configure the three-client completion job with `FedAvgRecipe`. |
| `evaluate.py` | Select `alpha` on validation summaries, then evaluate once on test data. |
| `run_demo.py` | Orchestrate data, optional Qwen LoRA FL, completion FL, and reporting. |

Data are prepared automatically by `run_demo.py`. To prepare them separately:

```bash
python prepare_data.py \
  --output-dir data/recoverable \
  --dataset-root ~/.cache/nvflare/fedcore \
  --scenario recoverable \
  --proxy-strength 0.9 \
  --seed 7
```

## Setup

Quick mode uses PyTorch SDPA and does not require FlashAttention:

```bash
cd research/fedcore
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default model is [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). Review its
model license before use. The first run downloads model weights from Hugging Face and MNIST through `torchvision`.

## One-GPU quickstart

The quickstart uses the frozen Qwen3-VL predictor, extracts all site features sequentially on one GPU, and runs the
small completion clients on CPU:

```bash
python run_demo.py --mode quick --scenario recoverable --gpu 0
```

It is designed to finish in about 20 minutes on a modern data-center GPU, although model download and hardware speed
affect runtime. Run the cross-modal control with:

```bash
python run_demo.py --mode quick --scenario uninformative --gpu 0
```

## Expected behavior

- `recoverable`: validation should select a nonzero `alpha`, and missing-image performance should improve without
  reducing aggregate validation performance.
- `uninformative`: validation should select `alpha=0`, retaining the image-missing baseline.
- `site-3`: every round should report `sent_empty_update=true` because the client has no paired image supervision.

Exact Qwen metrics depend on the model and runtime. These outcomes test the mechanics of FedCoRe rather than establish
a benchmark result.

## Optional federated Qwen LoRA mode

Full mode first invokes the existing [`examples/advanced/qwen3-vl`](../../examples/advanced/qwen3-vl/README.md)
training stack. It federates Qwen LoRA adapters, freezes the resulting global predictor, and then runs the same
completion workflow. Install the full dependency set and use one GPU per client:

```bash
bash ../../examples/advanced/qwen3-vl/install_requirements.sh "$(pwd)/requirements-full.txt"
python run_demo.py --mode full --scenario recoverable --gpu "[0],[1],[2]"
```

Use `--predictor-rounds` and `--predictor-max-steps` to change the optional predictor budget. Full mode reuses the
upstream Qwen3-VL implementation and attribution rather than maintaining a second copy here. Both image-present and
image-missing clients retain their locally available OCR report during predictor training.

The installer deliberately installs PyTorch before building `flash_attn` with `--no-build-isolation`; a compatible
CUDA toolkit and compiler must be available.

## Outputs

By default, runs write to `outputs/<scenario>_seed<seed>/`:

```text
data/                          local manifests and rendered MNIST images
feature_cache/site-*/         local Qwen caches
completion/site-*/            per-round valid-supervision metrics
completion/global_model.pt    final federated completion checkpoint
evaluation/summary.json       aggregate selection and evaluation report
evaluation/per_site_metrics.json
run_config.json               resolved command configuration
```

The reusable torchvision MNIST download is stored under `--dataset-root`, which defaults to
`~/.cache/nvflare/fedcore`. `run_demo.py` places the NVFlare workspace under the run's output directory by default,
so full-mode checkpoints use persistent project storage rather than `/tmp`. Direct `job.py` use retains its own
`/tmp/nvflare/fedcore` default. Use `--workspace` to select a larger filesystem. Both output and workspace paths
must be fresh; existing paths are rejected to prevent concurrent runs or stale checkpoints from being mixed.

`summary.json` reports image-present performance on paired examples, naturally image-missing performance before and
after completion, aggregate policy performance, contributing clients and paired-example counts, the selected `alpha`,
per-client validation sufficient statistics, and the full validation candidate table. Test caches are opened only
after validation selection completes.

Pooled AUROC is computed locally by this single-machine simulator for tutorial verification. It is not a
privacy-preserving production aggregation protocol.

## Adapting to another modality or dataset

The cache contract is target-modality neutral. Replace the MNIST preparation and feature extraction while preserving
these fields per site and split:

- `example_ids`, `labels`, and `image_available` (rename semantically in your loader if needed);
- `missing_features` and `missing_logits` for every record;
- `paired_mask`, equal to the boolean availability mask, and `full_logits`, finite for paired examples and `NaN`
  elsewhere.

Cache files are loaded with PyTorch's restricted `weights_only=True` deserializer. Store numeric arrays as
`torch.Tensor` values and use only built-in Python containers for metadata; convert NumPy arrays before `torch.save`.
Labels must be binary integer tensors, masks must be boolean, and feature/logit tensors must follow the finite/`NaN`
contract above. Unsupported or corrupt serialized objects fail before training.

The completion job then remains unchanged. Real deployments should generate caches at each site, use an appropriate
federated validation metric, and define modality masks from their actual acquisition workflow.

## Privacy and limitations

Rendered digit images and OCR reports remain in site-specific directories during training, and clients exchange only
completion parameters. The single-machine tutorial evaluator can read all site caches to report pooled AUROC; a
production deployment should replace that diagnostic with an approved federated evaluation protocol. This workflow
does not provide a formal privacy guarantee: model updates can leak information without additional controls such as
secure aggregation or differential privacy.

The OCR sensor is intentionally controllable and does not model clinical missingness, institutional shift, or the
difficulty of recovering a patient-specific clinical modality. Do not interpret tutorial gains as evidence for
clinical deployment.

## License

The code in this directory follows NVFlare's Apache License 2.0; see the [repository license](../../LICENSE).
MNIST is downloaded separately through [`torchvision.datasets.MNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html);
review the dataset's source terms before use. Qwen3-VL model weights are downloaded separately and have their own
terms; review the [model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). Optional full mode reuses the upstream
example, whose vendored Qwen training code is attributed in its
[NOTICE](../../examples/advanced/qwen3-vl/NOTICE).

## Citation

Holger R. Roth, Ziyue Xu, and Peter Cnudde, "[FedCoRe: Target-Adaptive Completion for Missing Modalities in Healthcare Federated Learning](https://arxiv.org/abs/2608.18311)," arXiv:2608.18311, 2026.
