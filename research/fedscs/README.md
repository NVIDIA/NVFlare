# FedSCS: Robust Federated Learning via Stable Cosine Similarity

🏆 Award: This work received the Distinguished Conference Paper Award at the IEEE ICCST 2025.

FedSCS is a robust federated aggregation method that assigns adaptive weights to client updates using **peer-update similarity** and **temporal stability**. It is intended for federated learning settings with heterogeneous or potentially unreliable client updates.

## Motivation

Standard FedAvg primarily weights clients according to local dataset size. When client data are non-IID or a client update is substantially different from the updates of other clients, dataset-size weighting alone may assign substantial influence to an update that is inconsistent with the prevailing update direction.

FedSCS addresses this by comparing each client update with the aggregate direction of its peers and tracking the client's similarity across communication rounds. Updates that are consistently aligned with their peers receive higher aggregation weights.

FedSCS does **not** require access to clients' raw training data and should not be interpreted as providing guaranteed malicious-client detection.

## FedSCS

For client \(i\) at round \(t\), FedSCS:

1. Computes the peer-consensus update from the other participating clients.
2. Computes the non-negative cosine similarity between the client update and peer consensus.
3. Maintains a rolling similarity score across rounds.
4. Penalizes rapidly changing similarity through a stability term.
5. Normalizes the resulting scores into aggregation weights.
6. Aggregates client updates using these weights.

The method requires \(O(Nd)\) operations for \(N\) participating clients and update dimension \(d\), without constructing a pairwise client-similarity matrix.

## NVIDIA FLARE Integration

This example implements FedSCS as a **custom NVIDIA FLARE model aggregator** and uses the standard `FedAvgRecipe` for the federated workflow.

```text
FedAvgRecipe
     │
     ├── Client 1 ──┐
     ├── Client 2 ──┤
     ├── Client 3 ──┤── DIFF updates
     ├── Client 4 ──┤
     └── Client 5 ──┘
                    │
             FedSCSAggregator
                    │
          Peer similarity + stability
                    │
             Adaptive weights
                    │
               Global model
```

Model updates are transferred using NVIDIA FLARE's `DIFF` transfer type.

## Example Dataset

The example uses CIFAR-10 with five simulated clients. The dataset is prepared locally and is not included in the repository.

The dataset preparation creates four regular clients and one intentionally noisy client (`site-5`). For `site-5`, 80% of the training samples are corrupted using salt-and-pepper noise, with a salt probability of 0.30 and a pepper probability of 0.30.

The standard clean CIFAR-10 test set is used for evaluation.

This setup provides a controlled example of heterogeneous client updates and allows FedSCS to evaluate client updates based on their similarity to the updates of the other participating clients.

## Update Safety

FedSCS uses cosine similarity, whose trust score is invariant to positive scalar rescaling of a client update. To provide defense in depth against arbitrarily large finite DIFF values, this implementation applies a configurable L2 norm bound to received client updates before they are used by the aggregator. The default example uses a maximum update norm of 10.0.

This magnitude bound is an implementation-level safety control and is separate from the Stable Cosine Similarity scoring formulation described in the FedSCS paper.

## Reproducibility and Ablation

To evaluate the effect of the FedSCS weighting mechanism separately from the update-norm safety bound, three matched aggregation conditions were evaluated:

1. **FedAvg** — standard NVIDIA FLARE FedAvg without update clipping.
2. **FedAvg + L2 clipping** — standard FedAvg with the same maximum client-update L2 norm of 10.0 used by the FedSCS implementation.
3. **FedSCS + L2 clipping** — FedSCS with the same maximum client-update L2 norm of 10.0.

All three conditions use the same CIFAR-10 client partitions, noisy `site-5` construction, training configuration, number of rounds, and seed-specific initial model checkpoints.

Five experiment seeds were evaluated:

```text
1001, 1002, 1003, 1004, 1005
```

For each seed, the final global model after the fixed 10-round training procedure was evaluated independently on the same clean 10,000-image CIFAR-10 test set.

### Final-model test accuracy

| Seed | FedAvg | FedAvg + L2 clipping | FedSCS + L2 clipping |
|------|-------:|---------------------:|---------------------:|
| 1001 | 64.94% | 64.57% | 68.07% |
| 1002 | 67.44% | 67.78% | 69.77% |
| 1003 | 66.23% | 65.78% | 68.30% |
| 1004 | 65.61% | 65.81% | 69.40% |
| 1005 | 66.71% | 67.19% | 68.84% |
| **Mean ± Std.** | **66.19 ± 0.97%** | **66.23 ± 1.27%** | **68.88 ± 0.72%** |

Across the five matched seeds, the mean paired difference between FedSCS and standard FedAvg was **+2.69 percentage points**. The mean paired difference between FedSCS and FedAvg with the same L2 clipping was **+2.65 percentage points**.

The corresponding mean difference between FedAvg with clipping and standard FedAvg was **+0.04 percentage points**.

### Best-checkpoint test accuracy

Best checkpoints selected during the federated run were evaluated separately on the same clean test set.

| Seed | FedAvg | FedAvg + L2 clipping | FedSCS + L2 clipping |
|------|-------:|---------------------:|---------------------:|
| 1001 | 67.32% | 67.28% | 69.05% |
| 1002 | 67.97% | 68.68% | 69.10% |
| 1003 | 65.84% | 65.39% | 67.61% |
| 1004 | 67.02% | 67.27% | 68.28% |
| 1005 | 67.48% | 66.44% | 68.77% |
| **Mean ± Std.** | **67.13 ± 0.80%** | **67.01 ± 1.21%** | **68.56 ± 0.62%** |

The final-model results are the primary reproducibility metric because all methods were trained for the same fixed number of communication rounds. Best-checkpoint results are provided separately to characterize model-selection performance.

## Running the Baselines

The job supports the following aggregation methods:

```bash
python research/fedscs/job.py --method fedavg
python research/fedscs/job.py --method fedavg_clipped
python research/fedscs/job.py --method fedscs
```

The default method is `fedscs`.

For reproducible experiments, provide the corresponding seed-specific initial checkpoint:

```bash
python research/fedscs/job.py \
    --method fedscs \
    --seed 1001 \
    --initial_ckpt /path/to/experiments/seed_1001/initial_model.pt
```

The same procedure can be used with `fedavg` and `fedavg_clipped`.

## Project Structure

```text
research/fedscs/
├── README.md
├── requirements.txt
├── job.py
├── client.py
├── prepare_data.sh
└── src/
    ├── fedscs_aggregator.py
    └── model.py
```

## Requirements

* Python 3.10+
* NVIDIA FLARE ~= 2.9.0
* PyTorch
* torchvision
* NumPy

Install the dependencies with:

```bash
pip install -r research/fedscs/requirements.txt
```

## Prepare CIFAR-10

From the NVIDIA FLARE repository root:

```bash
./research/fedscs/prepare_data.sh
```

The preparation script downloads CIFAR-10 and validates the required dataset files before reporting successful preparation.

## Run the Example

From the NVIDIA FLARE repository root:

```bash
cd research/fedscs
python job.py
```

The example runs a simulated federated learning experiment using the standard NVIDIA FLARE Recipe workflow and the custom `FedSCSAggregator`.

During training, the FedSCS aggregator reports the per-round client scores, Stable Cosine Similarity (SCS) values, and normalized aggregation weights.

In simulator runs, these per-client `score`, `scs`, and `weight` INFO messages are written to:

```text
<workspace>/server/log.txt
```

Inspect these values across rounds to see how FedSCS adapts the contribution of each client.

In particular, compare the aggregation weight assigned to `site-5` with the weights of the other clients. Because `site-5` contains intentionally noisy training data, its FedSCS weight reflects its similarity and stability relative to the other participating clients.

## Evaluation

The clean CIFAR-10 test set contains 10,000 images.

Saved global checkpoints can be evaluated independently using:

```bash
python research/fedscs/evaluate_checkpoint.py \
    --checkpoint /path/to/FL_global_model.pt \
    --test_data /path/to/test.pt
```

Both the final global checkpoint and the best selected checkpoint can be evaluated independently.

## Limitations

FedSCS relies on peer consensus as a reference for evaluating client updates. The example uses five clients, with four regular clients and one noisy client, and provides a controlled CIFAR-10 demonstration rather than covering all forms of non-IID data, client heterogeneity, data corruption, or adversarial behavior.

The reproducibility study uses five seeds and a fixed five-client CIFAR-10 configuration. The reported results should therefore be interpreted as evidence from this controlled experimental setting rather than as a characterization of all federated learning environments.

## Citation

If you use FedSCS in academic work, please cite:

```text
Rakib Ul Haque and Panagiotis (Panos P.) Markopoulos,
"Robust Federated Learning via Stable Cosine Similarity,"
IEEE ICCST, 2025.
```

## License

This research example follows the licensing terms of the NVIDIA FLARE repository.
