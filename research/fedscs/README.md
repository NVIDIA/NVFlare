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
     ├── Client 3 ──┤
     ├── Client 4 ──┤── DIFF updates
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

The dataset preparation creates four regular clients and one intentionally noisy client (`site-5`). For `site-5`, 80% of the training samples are corrupted using salt-and-pepper noise, with a salt probability of 0.30 and a pepper probability of 0.30. The standard clean CIFAR-10 test set is used for evaluation.

This setup provides a controlled example of heterogeneous client updates and allows FedSCS to evaluate client updates based on their similarity to the updates of the other participating clients.

## Update Safety

FedSCS uses cosine similarity, whose trust score is invariant to positive scalar rescaling of a client update. To provide defense in depth against arbitrarily large finite DIFF values, this implementation applies a configurable L2 norm bound to received client updates before they are used by the aggregator. The default example uses a maximum update norm of 10.0.

This magnitude bound is an implementation-level safety control and is separate from the Stable Cosine Similarity scoring formulation described in the FedSCS paper.

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

During training, the FedSCS aggregator reports the per-round client scores, Stable Cosine Similarity (SCS) values, and normalized aggregation weights in the NVIDIA FLARE log. Inspect these values across rounds to see how FedSCS adapts the contribution of each client.

In particular, compare the aggregation weight assigned to `site-5` with the weights of the other clients. Because `site-5` contains intentionally noisy training data, its FedSCS weight reflects its similarity and stability relative to the other participating clients.

## Limitations

FedSCS relies on peer consensus as a reference for evaluating client updates. The example uses five clients, with four regular clients and one noisy client, and provides a controlled CIFAR-10 demonstration rather than covering all forms of non-IID data, client heterogeneity, data corruption, or adversarial behavior.

## Citation

If you use FedSCS in academic work, please cite:

```text
Rakib Ul Haque and Panagiotis (Panos P.) Markopoulos,
"Robust Federated Learning via Stable Cosine Similarity,"
IEEE ICCST, 2025.
```

## License

This research example follows the licensing terms of the NVIDIA FLARE repository.
