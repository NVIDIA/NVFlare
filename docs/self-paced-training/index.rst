.. _self_paced_training:

Self-Paced-Training Tutorials
==============================

Federated Learning with NVIDIA FLARE: Notebooks and Videos
----------------------------------------------------------
Welcome to the five-part course on Federated Learning with NVIDIA FLARE!
This course covers everything from the fundamentals to advanced applications, system deployment, privacy, security, and real-world industry use cases.
It has **100+** notebooks and **80** videos.


What You'll Learn
-----------------

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

.. tip::

   While each notebook is self-contained and can be run independently, for best results, follow the sequence to build a strong foundation.

Course Outline
--------------

Part 1: Introduction to Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provides a hands-on introduction to federated learning using NVIDIA FLARE. You will learn how to run and develop federated learning applications, with practical examples and clear guidance for both beginners and experienced practitioners.

**What You'll Learn**

- The basics of federated learning and its advantages
- How to use NVIDIA FLARE to train and deploy federated learning models
- Transitioning from standard ML code to federated learning workflows
- Customizing client and server logic in NVIDIA FLARE
- Understanding job structure, configuration, and statistics

**Chapter 1: Running Federated Learning Applications**

- Train an image classification model with PyTorch using NVIDIA FLARE
- Convert standard PyTorch training code to federated learning code
- Customize client and server logic in NVIDIA FLARE
- Explore job structure and configuration for federated learning

.. tip::

    `Chapter 1 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-1_federated_learning_introduction/chapter-1_running_federated_learning_applications/01.0_introduction/introduction.ipynb>`_,
    `Chapter 1 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-1_federated_learning_introduction/chapter-1_running_federated_learning_applications/video.md>`_

**Chapter 2: Developing Federated Learning Applications**


- Perform federated statistics for both image and tabular data
- Convert PyTorch Lightning and traditional ML code to federated learning workflows with NVIDIA FLARE
- Use the NVIDIA FLARE Client API for advanced customization

.. tip::

    `Chapter 2 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-1_federated_learning_introduction/chapter-2_develop_federated_learning_applications/02.0_introduction/introduction.ipynb>`_ ,
    `Chapter 2 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-1_federated_learning_introduction/chapter-2_develop_federated_learning_applications/video.md>`_

Part 2: Federated Learning System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section explores the architecture and deployment of federated computing systems using NVIDIA FLARE. You will gain practical knowledge on system setup, user interaction, and monitoring tools for federated learning environments.

**What You'll Learn**

- NVIDIA FLARE system architecture and core concepts
- How to set up and simulate a federated computing system (local deployment)
- User interaction methods: admin console, Python API, and CLI
- Monitoring system events with Prometheus and Grafana

**Chapter 3: Federated Computing Platform**

- Understand the NVIDIA FLARE federated computing platform and its components
- Learn about system roles, communication, and workflow

.. tip::

    `Chapter 3 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-2_federated_learning_system/chapter-3_federated_computing_platform/03.0_introduction/introduction.ipynb>`_ ,
    `Chapter 3 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-2_federated_learning_system/chapter-3_federated_computing_platform/video.md>`_

**Chapter 4: Setup Federated Computing System**

- Step-by-step guide to setting up a federated computing system with NVIDIA FLARE
- Simulate deployments and interact with the system using various tools

.. tip::

    `Chapter 4 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-2_federated_learning_system/chapter-4_setup_federated_system/04.0_introduction/introduction.ipynb>`_ ,
    `Chapter 4 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-2_federated_learning_system/chapter-4_setup_federated_system/video.md>`_

Part 3: Security and Privacy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Federated learning enables decentralized model training while preserving data privacy, making it ideal for sensitive domains like healthcare and finance. However, federated learning introduces security and privacy risks, such as data leakage, adversarial attacks, and model integrity threats.

**What You'll Learn**

- Privacy risks and attack vectors in federated learning
- Protections: differential privacy, secure aggregation, homomorphic encryption
- Security challenges: adversarial attacks, unauthorized access, communication threats
- Security solutions: authentication, RBAC, encrypted communication, trust mechanisms
- How NVIDIA FLARE implements robust security and privacy for federated learning

**Chapter 5: Privacy in Federated Learning**

- Understand privacy risks and attacks in federated learning
- Explore privacy-preserving techniques with NVIDIA FLARE

.. tip::

    `Chapter 5 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-3_security_and_privacy/chapter-5_Privacy_In_Federated_Learning/05.0_introduction/introduction.ipynb>`_ ,
    `Chapter 5 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-3_security_and_privacy/chapter-5_Privacy_In_Federated_Learning/video.md>`_


**Chapter 6: Security in Federated Computing System**

- Learn about security threats and solutions in federated learning
- See how NVIDIA FLARE enforces secure communication, authentication, and access control

.. tip::

    `Chapter 6 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-3_security_and_privacy/chapter-6_Security_in_federated_compute_system/06.0_introduction/introduction.ipynb>`_ ,
    `Chapter 6 Video <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-3_security_and_privacy/chapter-6_Security_in_federated_compute_system/video.md>`_

Part 4: Advanced Topics in Federated Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section explores advanced topics and techniques in federated learning using NVIDIA FLARE. You will learn about cutting-edge algorithms, workflows, large language model (LLM) training, secure XGBoost, and the distinction between high-level and low-level APIs.

**What You'll Learn**

- Advanced federated learning algorithms: FedOpt, FedProx, and more
- Workflows: cyclic, split learning, swarm learning
- Training and fine-tuning large language models (LLMs) with NVIDIA FLARE
- Secure federated XGBoost
- High-level vs. low-level APIs in NVIDIA FLARE

**Chapter 7: Federated Learning Algorithms and Workflows**

- Explore various federated learning algorithms and workflow strategies with NVIDIA FLARE

.. tip::

    `Chapter 7 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-7_algorithms_and_workflows/07.0_introduction/introduction.ipynb>`_ ,
    `Chapter 7 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-7_algorithms_and_workflows/video.md>`_

**Chapter 8: Federated LLM Training**

- Learn how to train and fine-tune large language models in a federated setting with NVIDIA FLARE

.. tip::

    `Chapter 8 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-8_federated_LLM_training/08.0_introduction/introduction.ipynb>`_ ,
    `Chapter 8 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-8_federated_LLM_training/video.md>`_

**Chapter 9: NVIDIA FLARE Low-level APIs**

- Discover the power and flexibility of NVIDIA FLARE's low-level APIs

.. tip::

    `Chapter 9 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-9_flare_low_level_apis/09.0_introduction/introduction.ipynb>`_ ,
    `Chapter 9 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-9_flare_low_level_apis/video.md>`_

**Chapter 10: Federated XGBoost**

- Step-by-step guide to secure federated XGBoost with NVIDIA FLARE

.. tip::

    `Chapter 10 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-10_federated_XGBoost/10.0_introduction/introduction.ipynb>`_ ,
    `Chapter 10 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-4_advanced_federated_learning/chapter-10_federated_XGBoost/video.md>`_

Part 5: Federated Learning Applications in Industries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section demonstrates how NVIDIA FLARE is applied in real-world industry settings, focusing on healthcare, life sciences, and financial services. Learn how federated learning enables collaboration, privacy, and innovation across organizations.

**What You'll Learn**

- How NVIDIA FLARE powers collaborative machine learning in healthcare and life sciences, including:

  - Medical image analysis (e.g., cancer detection, radiology)
  - Survival analysis (e.g., Kaplan-Meier)
  - Genomics and multi-institutional research
  - Drug discovery

- Financial services applications, such as:

  - Fraud detection
  - Anomaly detection in transactions

**Chapter 11: Federated Learning in Healthcare and Life Sciences**

- Use cases for NVIDIA FLARE in medical research, diagnostics, and drug discovery
- How to train robust, privacy-preserving models across hospitals and research centers

.. tip::

    `Chapter 11 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-5_federated_learning_applications_in_industries/chapter-11_federated_learning_in_healthcare_lifescience/11.0_introduction/introduction.ipynb>`_ ,
    `Chapter 11 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-5_federated_learning_applications_in_industries/chapter-11_federated_learning_in_healthcare_lifescience/video.md>`_


**Chapter 12: Federated Learning in Financial Services**

- Collaborative model training for fraud detection, credit risk, and regulatory compliance

.. tip::

    `Chapter 12 Notebooks <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-5_federated_learning_applications_in_industries/chapter-12_federated_learning_in_financial_services/12.0_introduction/introduction.ipynb>`_ ,
    `Chapter 12 Videos <https://github.com/NVIDIA/NVFlare/blob/main/examples/tutorials/self-paced-training/part-5_federated_learning_applications_in_industries/chapter-12_federated_learning_in_financial_services/video.md>`_


Getting Started
---------------

- Start with any part or topic of interest, or follow the sequence for a comprehensive journey.
- Refer to the official `NVIDIA FLARE documentation <https://nvflare.readthedocs.io/>`_ for deeper dives and troubleshooting.

Happy learning!