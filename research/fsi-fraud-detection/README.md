# Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE

Implementation for the paper [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617).

Holger R. Roth, Sarthak Tickoo, Mayank Kumar, Isaac Yang, Andrew Liu, Amit Varshney, Sayani Kundu, Iustina Vintila, Peter Madsgaard, Juraj Milcak, Chester Chen, Yan Cheng, Andrew Feng, Jeff Savio, Vikram Singh, Craig Stancill, Gloria Wan, Evan Powell, Anwar Ul Haq, Sudhir Upadhyay, Jisoo Lee

> Fraud-related financial losses continue to rise, while regulatory, privacy, and data-sovereignty constraints increasingly limit the feasibility of centralized fraud detection systems. Federated Learning (FL) has emerged as a promising paradigm for enabling collaborative model training across institutions without sharing raw transaction data. Yet, its practical effectiveness under realistic, non-IID financial data distributions remains insufficiently validated.
>
> In this work, we present a multi-institution, industry-oriented proof-of-concept study evaluating federated anomaly detection for payment transactions using the NVIDIA FLARE framework. We simulate a realistic federation of heterogeneous financial institutions, each observing distinct fraud typologies and operating under strict data isolation. Using a deep neural network trained via federated averaging (FedAvg), we demonstrate that federated models achieve a mean F1-score of 0.903—substantially outperforming locally trained models (0.643) and closely approaching centralized training performance (0.925), while preserving full data sovereignty.
>
> We further analyze convergence behavior, showing that strong performance is achieved within 10 federated communication rounds, highlighting the operational viability of FL in latency- and cost-sensitive financial environments. To support deployment in regulated settings, we evaluate model interpretability using Shapley-based feature attribution and confirm that federated models rely on semantically coherent, domain-relevant decision signals. Finally, we incorporate sample-level differential privacy via DP-SGD and demonstrate favorable privacy–utility trade-offs, achieving effective privacy budgets below $\epsilon = 10.0$ with moderate degradation in fraud detection performance. Collectively, these results provide empirical evidence that FL can enable effective cross-institution fraud detection, delivering near-centralized performance while maintaining strict data isolation and supporting formal privacy guarantees.

## Paper

- [Privacy-Preserving Federated Fraud Detection in Payment Transactions with NVIDIA FLARE](https://arxiv.org/abs/2603.13617)

## Highlights

- Studies payment fraud detection across heterogeneous financial institutions with non-IID transaction distributions.
- Evaluates federated anomaly detection with NVIDIA FLARE using federated averaging.
- Reports strong performance relative to local training while preserving data sovereignty.
- Explores interpretability with Shapley-based feature attribution and privacy-utility trade-offs with DP-SGD.

## Code

Code will be provided soon!
