.. _researcher_area:

####################
Researcher Area
####################

FedSM
======
The FedSM example illustrates the personalized federated learning algorithm FedSM accepted to CVPR 2022. It bridges the different data distributions across clients via a SoftPull mechanism and utilizes a Super Model. A model selector is trained to predict the belongings of a particular sample to any of the clients’ personalized models or global model. The training of this model also illustrates a challenging federated learning scenario with extreme label-imbalance, where each local training is only based on a single label towards the optimization for classification of a number of classes equivalent to the number of clients. In this case, the higher-order moments of the Adam optimizer are also averaged and synced together with model updates.

Auto-FedRL
===========
The Auto-FedRL example implements the automated machine learning solution described in Auto-FedRL: Federated Hyperparameter Optimization for Multi-institutional Medical Image Segmentation accepted to ECCV 2022. Conventional hyperparameter optimization algorithms are often impractical in real-world FL applications as they involve numerous training trials, which are often not affordable with limited computing budgets. Auto-FedRL proposes an efficient reinforcement learning (RL)-based federated hyperparameter optimization algorithm, in which an online RL agent can dynamically adjust the hyperparameters of each client based on the current training progress.

Quantifying Data Leakage in Federated Learning
===============================================
This research example contains the tools necessary to recreate the chest X-ray experiments described in Do Gradient Inversion Attacks Make Federated Learning Unsafe?, accepted to IEEE Transactions on Medical Imaging. It presents new ways to measure and visualize potential data leakage in FL using a new FLARE filter that can quantify the data leakage for each client and visualize it as a function of the FL training rounds. Quantifying the data leakage in FL can help determine the optimal tradeoffs between privacy-preserving techniques, such as differential privacy, and model accuracy based on quantifiable metrics.