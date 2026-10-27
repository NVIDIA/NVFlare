# Federated Learning with NVIDIA FLARE

Welcome to the five-part course on Federated Learning with NVIDIA FLARE!  
This course covers everything from the fundamentals to advanced applications, system deployment, privacy, security, and real-world industry use cases.

> **NVFlare 2.10 course update:** [Running Federated Learning Applications](part-1_federated_learning_introduction/chapter-1_running_federated_learning_applications/README.md) now uses the Recipe API and has been validated against the 2.10 development branch. Other sections are being migrated and reviewed incrementally; consult each section's README for its validation status. The [archived 2.6 course](https://github.com/NVIDIA/NVFlare/tree/2.6.0/examples/tutorials/self-paced-training) remains available for the original notebooks and videos.

---

## What You'll Learn

- **Fundamentals:**  
  Understand federated learning and decentralized training concepts.
- **System Architecture:**  
  Learn about NVIDIA FLARE system architecture, deployment, and user interactions.
- **Privacy & Security:**  
  Explore privacy and security challenges, solutions, and enterprise-grade protections.
- **Advanced Topics:**  
  Dive into algorithms (FedOpt, FedProx, etc.), workflows (cyclic, split, swarm), LLM training, and XGBoost.
- **Industry Applications:**  
  Discover real-world use in healthcare, life sciences, and finance.
- **Practical Skills:**  
  Transition from standard ML code to federated workflows; customize client/server logic, job structure, and configuration.
- **Comprehensive Resources:**  
  Access over 100 notebooks and 88 videos for a thorough learning experience.

> **Tip:**  
> Follow each chapter's setup and prerequisites. Each revised hands-on application lesson has its own runnable code and variation beside the notebook. Follow the learning sequence, or open a section independently after completing its prerequisites.

---

## Course Outline

### [Introduction to Federated Learning](./part-1_federated_learning_introduction/part-1_introduction.ipynb)

This section provides a hands-on introduction to federated learning using NVIDIA FLARE. You will learn how to run and develop federated learning applications, with practical examples and clear guidance for both beginners and experienced practitioners.

#### What You'll Learn

- The basics of federated learning and its advantages
- How to use NVIDIA FLARE to train and deploy federated learning models
- Transitioning from standard ML code to federated learning workflows
- Customizing client and server logic in NVIDIA FLARE
- Understanding job structure, configuration, and statistics

#### Running Federated Learning Applications [[Open notebooks](./part-1_federated_learning_introduction/chapter-1_running_federated_learning_applications/01.0_introduction/introduction.ipynb)]
- Run a CPU-only PyTorch application with `FedAvgRecipe` and `SimEnv`
- Convert local PyTorch training with the Client API
- Customize server settings and per-site training through recipe helpers
- Track metrics, evaluate the saved global model, and inspect recipe exports

#### Developing Federated Learning Applications [[Open notebooks](./part-1_federated_learning_introduction/chapter-2_develop_federated_learning_applications/02.0_introduction/introduction.ipynb)]
- Perform federated statistics for both image and tabular data
- Convert PyTorch Lightning and traditional ML code to federated learning workflows with NVIDIA FLARE
- Use the NVIDIA FLARE Client API for advanced customization

### [Federated Learning System](./part-2_federated_learning_system/part-2_introduction.ipynb)

This section explores the architecture and deployment of federated computing systems using NVIDIA FLARE. You will gain practical knowledge on system setup, user interaction, and monitoring tools for federated learning environments.

#### What You'll Learn

- NVIDIA FLARE system architecture and core concepts
- How to set up and simulate a federated computing system (local deployment)
- User interaction methods: admin console, Python API, and CLI
- Monitoring system events with Prometheus and Grafana

#### Federated Computing Platform [[Open notebooks](./part-2_federated_learning_system/chapter-3_federated_computing_platform/03.0_introduction/introduction.ipynb)]
- Understand the NVIDIA FLARE federated computing platform and its components
- Learn about system roles, communication, and workflow

#### Setup Federated Computing System [[Open notebooks](./part-2_federated_learning_system/chapter-4_setup_federated_system/04.0_introduction/introduction.ipynb)]
- Step-by-step guide to setting up a federated computing system with NVIDIA FLARE
- Simulate deployments and interact with the system using various tools

### [Security and Privacy](./part-3_security_and_privacy/part-3_introduction.ipynb)

Federated Learning (FL) enables decentralized model training while preserving data privacy, making it ideal for sensitive domains like healthcare and finance. However, FL introduces security and privacy risks, such as data leakage, adversarial attacks, and model integrity threats.

#### What You'll Learn

- Privacy risks and attack vectors in federated learning
- Protections: differential privacy, secure aggregation, homomorphic encryption
- Security challenges: adversarial attacks, unauthorized access, communication threats
- Security solutions: authentication, RBAC, encrypted communication, trust mechanisms
- How NVIDIA FLARE implements robust security and privacy for federated learning
 
#### Privacy in Federated Learning [[Open notebooks](./part-3_security_and_privacy/chapter-5_Privacy_In_Federated_Learning/05.0_introduction/introduction.ipynb)]
- Understand privacy risks and attacks in federated learning
- Explore privacy-preserving techniques with NVIDIA FLARE

#### Security in Federated Computing System [[Open notebooks](./part-3_security_and_privacy/chapter-6_Security_in_federated_compute_system/06.0_introduction/introduction.ipynb)]
- Learn about security threats and solutions in federated learning
- See how NVIDIA FLARE enforces secure communication, authentication, and access control

### [Advanced Topics in Federated Learning](./part-4_advanced_federated_learning/part-4_introduction.ipynb)

This section explores advanced topics and techniques in federated learning using NVIDIA FLARE. You will learn about cutting-edge algorithms, workflows, large language model (LLM) training, secure XGBoost, and the distinction between high-level and low-level APIs.

#### What You'll Learn

- Advanced federated learning algorithms: FedOpt, FedProx, and more
- Workflows: cyclic, split learning, swarm learning
- Training and fine-tuning large language models (LLMs) with NVIDIA FLARE
- Secure federated XGBoost
- High-level vs. low-level APIs in NVIDIA FLARE

#### Federated Learning Algorithms and Workflows [[Open notebooks](./part-4_advanced_federated_learning/chapter-7_algorithms_and_workflows/07.0_introduction/introduction.ipynb)]
- Explore various federated learning algorithms and workflow strategies with NVIDIA FLARE

#### Federated LLM Training [[Open notebooks](./part-4_advanced_federated_learning/chapter-8_federated_LLM_training/08.0_introduction/introduction.ipynb)]
- Learn how to train and fine-tune large language models in a federated setting with NVIDIA FLARE

#### NVIDIA FLARE Low-level APIs [[Open notebooks](./part-4_advanced_federated_learning/chapter-9_flare_low_level_apis/09.0_introduction/introduction.ipynb)]
- Discover the power and flexibility of NVIDIA FLARE's low-level APIs

#### Federated XGBoost [[Open notebooks](./part-4_advanced_federated_learning/chapter-10_federated_XGBoost/10.0_introduction/introduction.ipynb)]
- Step-by-step guide to secure federated XGBoost with NVIDIA FLARE

### [Federated Learning Applications in Industries](./part-5_federated_learning_applications_in_industries/part-5_introduction.ipynb)
This section demonstrates how NVIDIA FLARE is applied in real-world industry settings, focusing on healthcare, life sciences, and financial services. Learn how federated learning enables collaboration, privacy, and innovation across organizations.

#### What You'll Learn

- How NVIDIA FLARE powers collaborative machine learning in healthcare and life sciences, including:
  - Medical image analysis (e.g., cancer detection, radiology)
  - Survival analysis (e.g., Kaplan-Meier)
  - Genomics and multi-institutional research
  - Drug discovery (if specifically covered)
- Financial services applications, such as:
  - Fraud detection
  - Anomaly detection in transactions
 

#### Federated Learning in Healthcare and Life Sciences [[Open notebooks](./part-5_federated_learning_applications_in_industries/chapter-11_federated_learning_in_healthcare_lifescience/11.0_introduction/introduction.ipynb)]
- Use cases for NVIDIA FLARE in medical research, diagnostics, and drug discovery
- How to train robust, privacy-preserving models across hospitals and research centers

#### Federated Learning in Financial Services [[Open notebooks](./part-5_federated_learning_applications_in_industries/chapter-12_federated_learning_in_financial_services/12.0_introduction/introduction.ipynb)]
- Collaborative model training for fraud detection, credit risk, and regulatory compliance

---

## Getting Started

- Start with the validated [Running Federated Learning Applications guide](part-1_federated_learning_introduction/chapter-1_running_federated_learning_applications/README.md) for the Recipe API and 2.10 environment setup.
- Check each other section's migration status before using its notebooks with 2.10.
- Refer to the official [NVIDIA FLARE documentation](https://nvflare.readthedocs.io/) for deeper dives and troubleshooting.

---

Happy learning!
