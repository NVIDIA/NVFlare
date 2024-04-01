# Simple ML/DL to FL transition with NVFlare

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

For those who are more curious to learn more different types of implementations for use cases, you can continue to read. 

# For Advanced Users: Client API with different implementations 
 
For the same Client API, we have several implementations for different needs

  1. In-process Client API: the client training script is in the same process as the NVFlare Client job process.
     Since training script and client job process both in the same process memory, it is a bit efficient and simpler configuration, here we leverage ```InProcessClientAPIExecutor``` executor.
     In many cases, you should choose Option 1 for efficiency and simplicity. 


  2. sub-process client API: the client training script is in a sub-process
     Since training script and NVFlare job are in separate process with ```ClientAPILauncherExecutor``` executor, we offer two mechanisms to communicate
     * the sub-process training script and client job process are communicated via CellPipe (default)
     * the sub-process training script and client job process are communicated via FilePipe

     If you need to have multi-GPU or distributed pytorch training, you should use Option 2.
 

Note: Avoid install TensorFlow and PyTorch on the same virtual environment due to library conflicts.
