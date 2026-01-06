.. _flare_overview:

#####################
NVIDIA FLARE Overview
#####################

**NVIDIA FLARE** (NVIDIA Federated Learning Application Runtime Environment) is a domain-agnostic, open-source,
extensible Python SDK that allows researchers, data scientists, and data engineers to adapt existing ML/DL and
compute workflows to a federated paradigm. With the FLARE platform, developers can create secure and privacy-preserving
solutions for decentralized data computing, facilitating distributed multi-party collaboration.

FLARE supports **end-to-end federated learning**—from local simulation to large-scale production deployment—for both
**cross-silo** (institutional) and **cross-device** (edge/mobile) scenarios.


Key Features
============

Open & Developer-Friendly
-------------------------

- Apache 2.0 licensed with rich APIs and tooling
- Data scientist-friendly APIs requiring minimal code changes
- Comprehensive documentation and examples

Enterprise-Scale & Production-Ready
-----------------------------------

- Mature, secure, and scalable architecture
- Battle-tested in healthcare, financial services, and autonomous vehicles
- Deployed in both cloud and on-premises environments

Flexible Deployment
-------------------

- Supports on-premises, cloud, and hybrid environments
- Multiple deployment options: sub-processes, Docker, Kubernetes, or HPC
- Cloud deployment CLI for AWS and Azure

Robust Networking & Communication
---------------------------------

- Multi-protocol support (gRPC, TCP, HTTP)
- TLS/mTLS security with single-port operation
- LLM streaming and large data transfer capabilities
- Bring Your Own Connectivity (BYOConn) support

Framework & Model Agnostic
--------------------------

- Supports any ML framework: PyTorch, TensorFlow, scikit-learn, XGBoost, and more
- Works with any model type: LLMs, deep learning, traditional ML
- System-agnostic integration with various data processing frameworks

Strong Enterprise Security
--------------------------

- PKI-based authentication and authorization
- Role-based access control with local policy enforcement
- Secure provisioning with TLS certificates
- Comprehensive audit logging

Privacy & Compliance
--------------------

- Built-in differential privacy and homomorphic encryption
- Confidential computing with TEE support
- Multi-party Private Set Intersection (PSI)
- GDPR and HIPAA compliance support

Extensible Architecture
-----------------------

- Modular, event-based, and pluggable design
- Customizable components at every layer
- Easy integration with third-party systems via FLARE Agent

End-to-End Lifecycle
--------------------

- Complete workflow from research to production
- Consistent APIs across simulation, POC, and production modes
- Built-in support for LLM fine-tuning and distributed inference

Capabilities
============

Federated Computing
-------------------

At its core, FLARE is a federated computing framework upon which Federated Learning, Analytics, and
Evaluation are built. It is agnostic to datasets, workloads, and domains.

Unlike centralized data lake solutions that require copying data to a central location, FLARE brings
computing directly to distributed datasets. Data remains at each site, with only pre-approved results
shared among collaborators—ensuring data governance and privacy compliance.

Federated Training
------------------

Train models collaboratively across distributed data without centralizing sensitive information.

- **Models**: LLMs, deep learning, XGBoost, scikit-learn, PyTorch, TensorFlow, PyTorch Lightning
- **Workflows**: Federated averaging (FedAvg), swarm learning, cyclic training
- **Algorithms**: Horizontal FL, vertical FL, split learning
- **MLOps**: Real-time metrics streaming with TensorBoard, MLflow, and Weights & Biases

Federated Analytics
-------------------

Compute federated statistics across distributed datasets without direct data access.

- **Statistics**: Histograms, counts, means, min/max across distributed data
- **Data Exploration**: Privacy-preserving cohort discovery and feature analysis
- **Validation**: Cross-site data quality checks and schema validation

Federated Evaluation
--------------------

Assess model performance across distributed data without centralizing test datasets.

- **Model Evaluation**: Evaluate a global model across all participating clients
- **Cross-Site Evaluation**: Benchmark each client's model against data from other participants


Easy to Use
===========

FLARE provides intuitive APIs and tools that minimize the learning curve for data scientists and engineers.

FLARE Native APIs
-----------------

Convert existing ML code to federated learning with minimal changes.

- **Client API**: Add a few lines to existing training scripts—no FL expertise required
- **Job Recipe API**: Define complete FL jobs programmatically in Python
- **Collab API**: Simplified collaborative learning for common and advanced FL patterns

Flower-FLARE Integration
------------------------

Leverage the Flower ecosystem with FLARE's enterprise capabilities.

- **Native Execution**: Run existing Flower workflows in FLARE without code changes
- **Enhanced Features**: Add FLARE's metrics streaming, security, and scalability to Flower apps

Simulation & Deployment
-----------------------

Seamlessly transition from development to production with consistent APIs.

- **Simulator**: Rapid prototyping and debugging on a single machine
- **POC Mode**: Test federated workflows with realistic multi-process separation
- **Production**: Deploy to on-premises, cloud, or hybrid environments with full security


Industry Use Cases
==================

NVIDIA FLARE has been deployed across diverse industries worldwide.

**Healthcare & Life Sciences**

- Cancer research consortiums training tumor detection models across major medical centers
- Drug discovery collaborations among pharmaceutical companies using proprietary data
- Clinical trial recruitment, population genomics, and rare disease studies

**Financial Services**

- Fraud detection models trained across banking institutions
- Anti-money laundering (AML) with federated suspicious account detection
- Credit risk modeling with privacy-preserving data collaboration

**Scientific Computing**

- National laboratory platforms for scientific computing
- Federated Data mesh for weather prediction and climate research
- Research collaborations across institutional boundaries

**National Security**

- National laboratory platforms for large language model training
  under strict data governance and privacy compliance
- Closed-loop systems linking scientific discovery and national security initiatives

**Autonomous Systems**

- Cross-country autonomous vehicle model training
- EV battery range prediction and optimization
- Fleet-wide learning for transportation and logistics


Examples & Tutorials
====================

FLARE provides extensive built-in implementations and examples to accelerate development.

**Federated Training Workflows**

- Server-controlled: scatter-and-gather, cyclic weight transfer, federated evaluation
- Client-controlled: swarm learning, cross-site model evaluation
- Split learning: vertical partitioning for feature-distributed data

**Learning Algorithms**

- Aggregation: FedAvg, FedOpt, FedProx, SCAFFOLD
- Personalization: Ditto, FedSM, Fed AutoRL
- Advanced: Hierarchical FL, asynchronous FL (FedBuff)

**Privacy-Preserving Techniques**

- Homomorphic encryption for secure aggregation
- Differential privacy for gradient protection
- Multi-party Private Set Intersection (PSI)

**Domain Applications**

- LLM fine-tuning and distributed inference
- Medical imaging and healthcare AI
- Financial services (fraud detection, AML)
- Traditional ML (XGBoost, Random Forest, SVM, K-means)
- Graph neural networks and NLP

**Getting Started Tutorials**

- Step-by-step ML-to-FL conversion guides
- Simulator, POC mode, and production deployment
- Job Recipe API and Client API walkthrough

See :ref:`getting_started` and :ref:`tutorials` for comprehensive guides.


References
==========

For more detailed information, see:

- :ref:`flare_system_architecture` - Core system design and components
- :ref:`flare_security_overview` - Security architecture and features
- :ref:`client_api` - Client-side API for FL development
- :ref:`job_recipe` - Programmatic job definition
- :ref:`provisioning` - Secure deployment and provisioning
- :ref:`federated_statistics` - Federated analytics implementation
- :ref:`hello_pt` - Getting started with PyTorch examples
