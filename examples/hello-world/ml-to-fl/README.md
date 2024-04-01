# Convert Machine Learning/Deep Learning to Federated Learning transition with NVFlare

Converting Deep Learning (DL) to Federated Learning (FL) involves:
1. Algorithms formulation, how to formulate a DL to FL algorithm and what information needs to be passed between Client and Server
2. Convert existing standalone, centralized DL code to FL code.
3. Configure the workflow to use the newly changed code.

NVFlare make it easy to convert the existing machine learning (ML) or Deep Learning (DL) algorithm
to Federated Learning (FL) algorithm. As one can see, only a few lines of code changes.

In the [ml-to-fl](.) examples, we assume algorithm formulation is chosen from NVFLARE predefined workflow algorithms
(ex. FedAvg). You can find examples of converting traditional ML to FL in [step-by-step tutorials with tabular datasets](../step-by-step/higgs).

We will demonstrate different techniques depending on the existing code structure and preferences.

To configure the workflow, one can reference the config we have here and the documentation.

We cover the following use cases:

  1. Configurations of NVFlare Client API: [np](./np/README.md)
  2. PyTorch and PyTorch Lightning: [pt](./pt/README.md)
  3. TensorFlow: [tf](./tf/README.md)

If you just want to follow the steps to create convert the DL to FL code, you can just skip the rest of the readme and
go directly to the examples.  

For those who are more curious, you can continue to read. 

# For More Curious users
 
For the same Client API, we have several implementations for different needs

  1. In-process Client API: the client training script is in the same process as the NVFlare Client.
  2. sub-process client API: the client training script is in a sub-process
     * the sub-process training script and client process are communicated via FilePipe
     * the sub-process training script and client process are communicated via CellPipe 

  In most of the case, you should choose Option 1 for efficiency and simplicity.
  If you need to have multi-GPU or distributed pytorch training, you should use Option 2. For Option 2, we use CellPipe as default

To make this easy to follow, we have created two job templates: xxx-in-proc job templates are used for in-process client API. 


Note: Avoid install TensorFlow and PyTorch on the same virtual environment due to library conflicts.
