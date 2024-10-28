# NVFlare hello-world examples

This folder contains hello-world examples for NVFlare.

Please make sure you set up a virtual environment and install JupyterLab following the [example root readme](../README.md).

Please also install "./requirements.txt" in each example folder.

## Hello World Notebook
### Prerequisites
Before you run the notebook, the following preparation work must be done:

  1. Install a virtual environment following the instructions in [example root readme](../README.md)
  2. Install Jupyter Lab and install a new kernel for the virtualenv called `nvflare_example`
  3. Run [hw_pre_start.sh](./hw_pre_start.sh) in the terminal before running the notebook
  4. Run [hw_post_cleanup.sh](./hw_post_cleanup.sh) in the terminal after running the notebook 

* [Hello world notebook](./hello_world.ipynb)

## Hello World Examples
### Easier ML/DL to FL transition
* [ML to FL](./ml-to-fl/README.md): Showcases how to convert existing ML/DL code to an NVFlare job.

### Step by step examples
* [Step by step examples](./step-by-step/README.md): Shows specific techniques and workflows and what needs to be changed for each.

### Workflows
* [Hello Scatter and Gather](./hello-numpy-sag/README.md)
    * Example using [ScatterAndGather](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.scatter_and_gather.html) controller workflow.
* [Hello Cross-Site Validation](./hello-numpy-cross-val/README.md)
    * Example using [CrossSiteModelEval](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cross_site_model_eval.html) controller workflow.
* [Hello Cyclic Weight Transfer](./hello-cyclic/README.md)
    * Example using [CyclicController](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_common.workflows.cyclic_ctl.html) controller workflow to implement [Cyclic Weight Transfer](https://pubmed.ncbi.nlm.nih.gov/29617797/).
* [Hello Client Controlled Workflows](./hello-ccwf/README.md)
    * Example using [Client Controlled Workflows](https://nvflare.readthedocs.io/en/main/programming_guide/controllers/client_controlled_workflows.html).

### Deep Learning
* [Hello PyTorch](./hello-pt/README.md)
  * Example using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using ([FedAvg](https://arxiv.org/abs/1602.05629)) and [PyTorch](https://pytorch.org/) as the deep learning training framework.
* [Hello TensorFlow](./hello-tf2/README.md)
  * Example of using [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html) an image classifier using ([FedAvg](https://arxiv.org/abs/1602.05629)) and [TensorFlow](https://tensorflow.org/) as the deep learning training framework.
