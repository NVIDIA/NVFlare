###############
Getting Started
###############

This guide will help you get started with NVIDIA FLARE, a powerful platform for federated learning.

1. **Installation:**
   - Begin by visiting the :ref:`installation` section to set up NVFLARE on your system. This step is crucial to ensure that all necessary components are correctly installed.

2. **Running Modes:**
   - NVFLARE offers various modes to suit different needs, from development to production. Here’s an overview of the available modes:

   **Modes to Run NVFLARE**
   =========================
   NVFLARE supports different running modes to accommodate various use cases:

   .. list-table:: NVIDIA FLARE Modes
      :header-rows: 1

      * - **Mode**
        - **Documentation**
        - **Description**
      * - Simulator
        - :ref:`fl_simulator`
        - | The FL Simulator is a lightweight simulation tool that automates job runs on a 
          | single system. It is ideal for quickly running jobs or experimenting with research 
          | and FL algorithms.
      * - POC
        - :ref:`poc_command`
        - | POC mode simulates deployment on a single host. It uses separate processes for Clients and Server, 
          | allowing you to test the "provision" process locally with all pointing to localhost. 
          | You can also interact with the FLARE system locally via POC mode.
      * - Production
        - :ref:`provisioned_setup`
        - | The production mode involves a distributed deployment with startup kits generated 
          | from the provisioning process. It provides tools for provisioning, a dashboard, and 
          | various deployment options.

3. **Quick Start:**
   - In the :ref:`quickstart` section, we introduce a set of hello-world programs using various machine learning frameworks. 
     These examples primarily run in simulation or POC modes, guiding you through the initial steps 
     and helping you become familiar with the platform.

4. **Learn More:**
   - Now that you understand the different ways to run NVFLARE and have tried different quick start examples, 
     consider the following to learn more:

   * Watch videos for a quick taste of FL programming with FLARE for deep learning and traditional ML: 
   `<https://nvidia.github.io/NVFlare/>`_
    We present both videos and code for two examples: CIFAR10 classification and Kaplan-Meier survival analysis.
    The CIFAR10 example is demonstrated using PyTorch, Torch Lightning, and TensorFlow with Client Side, Server Side, and Job code.

   * Explore a huge number of examples in the tutorial `Catalog <https://nvidia.github.io/NVFlare/catalog/>`_
     Search and filter examples by experience level, framework, algorithm, application type, industry domain, API type, and privacy algorithm.
        
   * Dive into `step-by-step <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/step-by-step>` walk-throughs if you prefer to keep the same dataset while exploring different FL algorithms.
     The step-by-step examples essentially keep two datasets: CIFAR10 (image data) and Higgs dataset (Tabular data).
     For the CIFAR10 image dataset, we show you different FL algorithms, including FedAvg with scatter and gather type of workflow, Cyclic, swarm learning, Federated Statistics, and advanced features.
     For the Higgs dataset, we show you Fed Statistics, Scikit-learn, and XGBoost examples.

   * Dive into `ml-to-fl <https://github.com/NVIDIA/NVFlare/tree/main/examples/hello-world/ml-to-fl>` walk-throughs if you prefer to convert your standalone or centralized training code to FL code.
     The ml-to-fl examples focus on how to convert your standalone or centralized training code to FL code with different deep learning frameworks. 

  * Tutorials for different FLARE features:
    - How to use the POC command to set up and run FL jobs locally: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/setup_poc.ipynb>`
    - How to use the simulator CLI and in Python code: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/flare_simulator.ipynb>`
    - How to use the FLARE Python API to interact with the FLARE System, submit jobs, and monitor progress: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/flare_api.ipynb>`
    - Understand the FLARE logging configuration, output, and customize logging configuration: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/logging.ipynb>`
    - How to use the FLARE job CLI to submit jobs and modify the job configuration based on job templates: <https://github.com/NVIDIA/NVFlare/tree/main/examples/tutorials/job_cli.ipynb>`
  
5. **Research Topics:**
   - `Different FL research algorithms and implementations <https://github.com/NVIDIA/NVFlare/tree/main/research>`_

6. **Self-Paced Training**
   We offer extensive materials for self-paced training with NVIDIA FLARE, 
   including detailed tutorials and resources. For a comprehensive guide, 
   please refer to the **`Self-Paced Training <self-paced-training/self_paced_training.rst>`_** documentation. 
   This includes a 12-chapter course that provides an in-depth overview of FLARE, covering topics such as running federated 
   learning applications, algorithms, system architecture, experimental tracking, system monitoring, 
   and industrial applications. While each notebook can be run independently, it is recommended to follow 
   them sequentially for the best learning experience. With over 100 notebooks and 88 videos, this tutorial 
   serves as a thorough guide to federated learning with FLARE.

7. **Real-World Use Cases**:

   - `NVIDIA FLARE DAY Talks in 2024 <https://nvidia.github.io/NVFlare/flareDay/>`_
     Use cases from different industries, including healthcare, finance, and more.

   - NVIDIA FLARE DAY US+EMEA 2025: 2025-09-16 Workshop, 2025-09-17 Webinar : coming soon

   - NVIDIA FLARE DAY US+EMEA 2025: 2025-09-23 Workshop, 2025-09-24 Webinar : coming soon

8. **Discover FLARE via Chat**
  
   You can also use the chatbot to search the documentation, explore program flows, and find relevant code snippets, system architecture details, and more.

   Visit: [DeepWiki NVFlare Chatbot](https://deepwiki.com/NVIDIA/NVFlare)

   For example, you can ask the chatbot:
   - "Where can I set the heartbeat timeout between the client and server?"

   The chatbot will find all relevant locations and provide explanations for different types of heartbeat timeouts. You can follow up with questions like:

   - "I thought you could set the heartbeat in `resource.json` on the server or client side, which overwrites the default value."

   The chatbot will then give you more details on how to set the heartbeat timeout in `resource.json`.

   Other prompts you can ask include "How to fine-tune LLM," "What are the privacy features," etc.