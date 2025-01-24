# Simple ML/DL to FL transition with NVFlare


Converting Deep Learning (DL) models to Federated Learning (FL) entails several key steps:

Formulating the algorithm: This involves determining how to adapt a DL model into an FL framework, including specifying the information exchange protocol between the Client and Server.

Code conversion: Adapting existing standalone DL code into FL-compatible code. This typically involves minimal changes, often just a few lines of code, thanks to tools like NVFlare.

Workflow configuration: Once the code is modified, configuring the workflow to integrate the newly adapted FL code seamlessly.

NVFlare simplifies the process of transitioning from traditional Machine Learning (ML) or DL algorithms to FL. With NVFlare, the conversion process requires only minor code adjustments.

In our examples, we assume that algorithm formulation follows NVFlare's predefined workflow algorithms (such as FedAvg). Detailed tutorials on converting traditional ML to FL, particularly with tabular datasets, are available in our step-by-step guides.

We offer various techniques tailored to different code structures and user preferences. Configuration guidance and documentation are provided to facilitate workflow setup.

Our coverage includes:

Configurations for NVFlare Client API: [np](./np/README.md)
Integration with PyTorch and PyTorch Lightning frameworks:[pt](./pt/README.md)
Support for TensorFlow implementations: [tf](./tf/README.md)

For detailed instructions on configuring the workflow, refer to our provided examples and documentation.
If you're solely interested in converting DL to FL code, feel free to skip ahead to the examples without delving further into this readme.

For those eager to explore various implementations and use cases, read on.

## Advanced User Options: Client API with Different Implementations

Within the Client API, we offer multiple implementations tailored to diverse requirements:

* In-process Client API: In this setup, the client training script operates within the same process as the NVFlare Client job.
This configuration, utilizing the ```InProcessClientAPIExecutor```, offers shared the memory usage, is efficient and with simple configuration. 
Use this configuration for development or single GPU

* Sub-process Client API: Here, the client training script runs in a separate subprocess.
Utilizing the ```ClientAPILauncherExecutor```, this option offers flexibility in communication mechanisms:
  * Communication via CellPipe (default)
  * Communication via FilePipe ( no capability to stream experiment track log metrics) 
This configuration is ideal for scenarios requiring multi-GPU or distributed PyTorch training.

Choose the option best suited to your specific requirements and workflow preferences.

These implementations can be easily configured using the JobAPI's ScriptRunner.
By default, the ```InProcessClientAPIExecutor``` is used, however setting `launch_external_process=True` uses the ```ClientAPILauncherExecutor```
with pre-configured CellPipes for communication and metrics streaming.

Note: Avoid installing TensorFlow and PyTorch in the same virtual environment due to library conflicts.
