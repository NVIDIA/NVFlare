**************************
What's New in FLARE v2.6.0
**************************

Message Quantization
===================
Message quantization offers a solution to reduce communication overhead in federated learning by reducing the precision of transmitted updates. This feature is particularly beneficial for large language models (LLMs) where the default fp32 message precision can artificially inflate message size.

Key features:
  - Quantization and dequantization implemented with filter mechanism
  - No code changes required from user side - same training script works with/without quantization
  - Training and aggregation performed at original precision to minimize impact on training process
  - Support for both numpy arrays and torch Tensors
  - Direct cropping and casting for fp32 to fp16 conversion
  - 8- and 4-bit quantization using bitsandbytes

Message size comparison for a 1B parameter LLM:
  - 32-bit (fp32): 5716.26 MB (100%)
  - 16-bit (fp16, bf16): 2858.13 MB (50%)
  - 8-bit: 1429.06 MB + 1.54 MB meta (25.03%)
  - 4-bit (fp4, nf4): 714.53 MB + 89.33 MB meta (14.06%)

Native Tensor Transfer
=====================
FLARE 2.6.0 introduces support for native tensor transfer, allowing PyTorch tensors to be sent directly without serialization overhead. This eliminates the need for Tensor to Numpy conversion, preserving the original FPnn format. The feature is currently supported for PyTorch only.

Model Streaming Enhancements
===========================
Reduce Local Memory Usage
------------------------
The new object container streaming feature processes and transmits models incrementally, rather than requiring the entire dictionary of gradients to be stored in memory at once. This significantly reduces memory overhead for large models.

For example, a 70GB model with 1GB item-max:
  - Regular transmission: 70GB + 70GB = 140GB memory needed
  - Container streaming: 70GB + 1GB = 71GB memory needed

Support Unlimited Memory Streaming
--------------------------------
File-based streaming is introduced to handle models larger than available memory. This feature reads files chunk-by-chunk, requiring only enough memory to hold one chunk of data. The memory usage is independent of model size and only depends on file I/O settings.

Memory comparison for sending a 1B model:
  - Regular transmission: 42,427 MB peak memory, 47s completion time
  - Container streaming: 23,265 MB peak memory, 50s completion time
  - File streaming: 19,176 MB peak memory, 170s completion time

Note: Streaming enhancements are not yet integrated into high-level APIs or existing FL algorithm controllers/executors. Users can build custom controllers or executors to leverage this feature.

Structured Logging
=================
The structured logging feature addresses several customer concerns:
  - JSON format logging for data observability tools
  - Separation of training logs from communication logs
  - Dynamic log level changes for production debugging
  - Package-level hierarchy for granular control

Key improvements:
  - Changed from fileConfig to dictConfig
  - Default Logging Configuration file (log_config.json.default)
  - Dynamic Logging Configuration Commands
  - Multiple default log files:
    - log.txt: default log file
    - log.json: JSON format log
    - log_error.txt: ERROR level logs
    - log_fl.txt: FL task-specific logs
  - Predefined logging modes for simulator:
    - Concise: only FL tasks logs
    - Full: previous logging configuration
    - Verbose: debug level logging

Federated Statistics Extension
============================
Quantiles Support: Introduces quantile computation for federated statistics, helping summarize data distribution by providing key points that indicate how values are spread. Quantiles divide a probability distribution or dataset into intervals with equal probabilities, providing insights into data distribution patterns.

System Monitoring
================
FLARE Monitoring provides system metrics tracking for federated learning jobs, focusing on job and system lifecycle metrics. It leverages StatsD Exporter to monitor FLARE job and system events, which can be scraped by Prometheus and visualized with Grafana. This differs from machine learning experiment tracking by focusing on system-level metrics rather than training metrics.

Flower Integration v2
====================
NVFlare has been updated to work with the latest Flower system architecture, which separates the client app from the supernode process. This update enables more accurate job status information sharing between systems. Applications developed with Flower can run natively on the FLARE runtime without code modifications, combining Flower's design tools with FLARE's industrial-grade runtime.

HTTP Driver Enhancement
======================
The HTTP driver has been completely rewritten using aiohttp, significantly improving reliability and efficiency. The new implementation resolves previous issues with poor performance and network error recovery, matching the performance of GRPC and TCP drivers.

FLARE + BioNemo 2
================
NVFlare examples have been upgraded to use BioNeMo 2, enabling significant performance improvements on downstream tasks. The integrated BioNeMo ESM2 base models (650M) demonstrate notable gains in accuracy:

Subcellular Localization (SCL) Prediction:
  - BioNeMo 1: 0.773 accuracy
  - BioNeMo 2: 0.788 accuracy
  - FL: 0.776 to 0.817 accuracy improvement

New Features
===========
TensorBoard Metric Streaming Callback
------------------------------------
Implemented a callback for PyTorch Lightning to stream training metrics to the FL server via NVFlare, allowing real-time visualization of training curves.

Downstream Task Fitting
----------------------
Local Fine-Tuning tends to overfit, with training accuracy diverging from validation early. In contrast, Federated Averaging (FedAvg) models show continual performance improvement, highlighting the benefits of federated generalization over isolated training.

Tutorials and Education
======================
Self-paced-training tutorials covering:
  - Introduction to Federated Learning
  - Federated Learning System
  - Security and Privacy
  - Advanced Topics in Federated Learning
  - Federated Learning in Different Industries

New Examples
===========
1. Federated Embedding Model Training
2. Object Streaming
3. System Monitoring
4. NVIDIA FLARE on Google's FL Reference Architecture

**********************************
Migration to 2.6.0: Notes and Tips
**********************************


For PTClientAPILauncherExecutor and PTInProcessClientAPIExecutor
FLARE 2.6.0 introduces significant changes to the "params_exchange_format" argument in PTClientAPILauncherExecutor and PTInProcessClientAPIExecutor. These changes impact how data is exchanged between the client script and NVFlare.

### Changes in params_exchange_format
In previous versions, setting "params_exchange_format" to "pytorch" indicated that the client was using a PyTorch tensor on the third-party side. In this case, the tensor would be converted to a NumPy array before being sent back to NVFlare.

With the improvements introduced in FLARE 2.6.0, which now natively support PyTorch tensors during transmission, the meaning of "params_exchange_format" = "pytorch" has changed. Now, this setting directly sends PyTorch tensors to NVFlare without converting them to NumPy arrays.

### Action Required
To maintain the previous behavior (where PyTorch tensors are converted to NumPy arrays), you will need to explicitly set "params_exchange_format" to "numpy".

