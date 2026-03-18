# Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE

This research page references the paper [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617) and provides a stable entry in the NVIDIA FLARE research collection.

Holger R. Roth, Sarthak Tickoo, Mayank Kumar, Isaac Yang, Andrew Liu, Amit Varshney, Sayani Kundu, Iustina Vintila, Peter Madsgaard, Juraj Milcak, Chester Chen, Yan Cheng, Andrew Feng, Jeff Savio, Vikram Singh, Craig Stancill, Gloria Wan, Evan Powell, Anwar Ul Haq, Sudhir Upadhyay, Jisoo Lee

Financial fraud detection is often limited by privacy, regulatory, and data-sovereignty constraints that prevent institutions from pooling raw transaction data. This work evaluates a realistic multi-institution federated learning setup for payment fraud detection with NVIDIA FLARE and shows that federated training can substantially outperform local-only baselines while approaching centralized performance.

## Paper

- [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617)

## Highlights

- Studies payment fraud detection across heterogeneous financial institutions with non-IID transaction distributions.
- Evaluates federated anomaly detection with NVIDIA FLARE using federated averaging.
- Reports strong performance relative to local training while preserving data sovereignty.
- Explores interpretability with Shapley-based feature attribution and privacy-utility trade-offs with DP-SGD.

## Repository

This project is hosted in the NVIDIA FLARE repository under [`research/fsi-fraud-detection`](https://github.com/NVIDIA/NVFlare/tree/main/research/fsi-fraud-detection).
