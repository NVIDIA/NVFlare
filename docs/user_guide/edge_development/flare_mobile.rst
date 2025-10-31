.. _flare_mobile:

########################
FLARE Mobile Development
########################

FLARE 2.7 introduces comprehensive mobile development support for both Android and iOS platforms, enabling federated learning directly on edge devices. This guide covers mobile SDK integration, API usage, and best practices for developing FL applications on mobile platforms.

.. note::
   This guide assumes familiarity with the :ref:`edge development concepts <flare_edge>` and :ref:`hierarchical architecture <flare_hierarchical_architecture>`. For a complete understanding of the edge system, review the main :ref:`edge development guide <flare_edge>` first.

Overview
========

The FLARE Mobile SDK provides native libraries for Android (Kotlin/Java) and iOS (Swift/Objective-C) that enable:

* **On-device training**: Using ExecuTorch for mobile-optimized model execution
* **Federated learning integration**: With NVIDIA FLARE's hierarchical edge system
* **Real-time communication**: With FLARE servers via HTTP/HTTPS
* **Model management**: Including loading, training, and updating models
* **Data handling**: With flexible dataset interfaces
* **Error handling and recovery**: For mobile-specific scenarios

.. tip::
   For a quick start with mobile development, see the complete examples in `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_.

Platform Support
================

Android
-------
* **Minimum SDK**: API level 29 (Android 10)
* **Target SDK**: Latest stable
* **Language**: Kotlin/Java
* **Build System**: Gradle
* **Dependencies**: ExecuTorch, OkHttp, Gson, Coroutines

iOS
---
* **Minimum Version**: iOS 13.0
* **Target Version**: Latest stable
* **Language**: Swift/Objective-C
* **Build System**: Xcode
* **Dependencies**: ExecuTorch, Foundation, UIKit

Architecture
============

The Mobile SDK architecture consists of modular components including ``FlareRunner``, ``Connection``, ``DataSource``, ``ETTrainer``, and ``Dataset``. Each component is responsible for a specific aspect of federated learning on mobile devices, such as orchestration, communication, data handling, and model training. Refer to the component descriptions below for details.

Core Components
---------------

**FlareRunner** (Android: ``AndroidFlareRunner``, iOS: ``NVFlareRunner``)
    Main orchestrator that handles job fetching, task execution, and result reporting.

**Connection** (Android: ``Connection``, iOS: ``NVFlareConnection``)
    Manages HTTP/HTTPS communication with FLARE servers.

**DataSource** (Android: ``DataSource``, iOS: ``NVFlareDataSource``)
    Interface for providing training data to the FL system.

**ETTrainer** (Android: ``ETTrainer``, iOS: ``ETTrainer``)
    ExecuTorch-based trainer for on-device model training.

**Dataset** (Android: ``Dataset``, iOS: ``NVFlareDataset``)
    Data interface for feeding training examples to the trainer.

Getting Started
===============

Prerequisites
-------------

Before starting mobile development, ensure you have:

1. **NVIDIA FLARE Server**: A running FLARE server with hierarchical edge configuration (see :ref:`hierarchical architecture <flare_hierarchical_architecture>`)
2. **ExecuTorch**: Mobile-optimized PyTorch runtime (`ExecuTorch documentation <https://pytorch.org/executorch/>`_)
3. **Development Environment**: 
   * Android Studio (Android) - `Download <https://developer.android.com/studio>`_
   * Xcode (iOS) - Available from the Mac App Store
4. **Model**: A PyTorch model converted to ExecuTorch format
5. **Edge Examples**: Working examples in ``examples/advanced/edge/``

.. warning::
   ExecuTorch requires specific build configurations for mobile platforms. Ensure you follow the official ExecuTorch setup guide for your target platform.

Android Setup
=============

Installation
------------

1. **Add Dependencies** to your ``build.gradle.kts``:

.. code-block:: kotlin

   dependencies {
       // ExecuTorch dependencies
       implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
       implementation("com.facebook.soloader:nativeloader:0.10.5")
       implementation("com.facebook.fbjni:fbjni:0.5.1")
       
       // Network dependencies
       implementation("com.squareup.okhttp3:okhttp:4.12.0")
       implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")
       
       // JSON parsing
       implementation("com.google.code.gson:gson:2.10.1")
       
       // Coroutines for async operations
       implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
   }

2. **Copy SDK** to your project:

.. code-block:: bash

   cp -r nvflare/edge/device/android/sdk \
         app/src/main/java/com/nvidia/nvflare/

3. **Add ExecuTorch Libraries** to the ``app/libs/`` directory.

Basic Usage
-----------

.. code-block:: kotlin

   import com.nvidia.nvflare.sdk.core.AndroidFlareRunner
   import com.nvidia.nvflare.sdk.core.Connection
   import com.nvidia.nvflare.sdk.core.DataSource

   class MainActivity : AppCompatActivity() {
       private lateinit var flareRunner: AndroidFlareRunner
       
       override fun onCreate(savedInstanceState: Bundle?) {
           super.onCreate(savedInstanceState)
           
           // Create connection
           val connection = Connection(
               serverURL = "",  // Replace with your actual server URL
               allowSelfSignedCerts = true
           )
           
           // Create data source
           val dataSource = MyDataSource()
           
           // Create FlareRunner
           flareRunner = AndroidFlareRunner(
               context = this,
               connection = connection,
               jobName = "my_fl_job",
               dataSource = dataSource,
               deviceInfo = mapOf(
                   "device_id" to getDeviceId(),
                   "platform" to "android",
                   "app_version" to getAppVersion()
               ),
               userInfo = mapOf("user_id" to getUserId()),
               jobTimeout = 30.0f
           )
           
           // Start federated learning
           lifecycleScope.launch {
               flareRunner.run()
           }
       }
   }

iOS Setup
=========

Installation
------------

1. **Add ExecuTorch Framework** to your Xcode project.
2. **Copy NVFlareSDK** to your project:

.. code-block:: bash

   cp -r nvflare/edge/device/ios/NVFlareSDK YourProject/

3. **Add Framework** to your Xcode project target.

Basic Usage
-----------

.. code-block:: swift

   import NVFlareSDK
   import UIKit

   class ViewController: UIViewController {
       private var flareRunner: NVFlareRunner?
       
       override func viewDidLoad() {
           super.viewDidLoad()
           
           // Create data source
           let dataSource = MyDataSource()
           
           // Create FlareRunner
           flareRunner = try? NVFlareRunner(
               jobName: "my_fl_job",
               dataSource: dataSource,
               deviceInfo: [
                   "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
                   "platform": "ios",
                   "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
               ],
               userInfo: [:],
               jobTimeout: 30.0,
               serverURL: "",  // Replace with your actual server URL
               allowSelfSignedCerts: true
           )
           
           // Start federated learning
           Task {
               await flareRunner?.run()
           }
       }
   }

API Reference
=============

AndroidFlareRunner
------------------

The main orchestrator for Android federated learning.

**Constructor**

.. code-block:: kotlin

   AndroidFlareRunner(
       context: AndroidContext,
       connection: Connection,
       jobName: String,
       dataSource: DataSource,
       deviceInfo: Map<String, String>,
       userInfo: Map<String, String>,
       jobTimeout: Float,
       inFilters: List<Filter>? = null,
       outFilters: List<Filter>? = null,
       resolverRegistry: Map<String, Class<*>>? = null
   )

**Parameters**

- ``context``: Android application context.
- ``connection``: Connection instance for server communication.
- ``jobName``: Name of the FL job to participate in.
- ``dataSource``: Data source providing training data.
- ``deviceInfo``: Device metadata (``device_id``, ``platform``, etc.).
- ``userInfo``: User metadata (``user_id``, etc.).
- ``jobTimeout``: Timeout in seconds for job operations.
- ``inFilters``: Optional input filters for data processing.
- ``outFilters``: Optional output filters for result processing.
- ``resolverRegistry``: Optional component resolver registry.

**Methods**

.. code-block:: kotlin

   // Start federated learning
   suspend fun run()
   
   // Stop federated learning
   fun stop()
   
   // Get current status
   fun getStatus(): String

For more on android sdk API: check :ref:`mobile_android_api`.


NVFlareRunner (iOS)
-------------------

The main orchestrator for iOS federated learning.

**Initializer**

.. code-block:: swift

   init(
       jobName: String,
       dataSource: NVFlareDataSource,
       deviceInfo: [String: String],
       userInfo: [String: String],
       jobTimeout: TimeInterval,
       serverURL: String,
       allowSelfSignedCerts: Bool = false,
       inFilters: [NVFlareFilter]? = nil,
       outFilters: [NVFlareFilter]? = nil,
       resolverRegistry: [String: ComponentCreator.Type]? = nil
   ) throws

**Parameters**

- ``jobName``: Name of the FL job to participate in.
- ``dataSource``: Data source providing training data.
- ``deviceInfo``: Device metadata (``device_id``, ``platform``, etc.).
- ``userInfo``: User metadata (``user_id``, etc.).
- ``jobTimeout``: Timeout in seconds for job operations.
- ``serverURL``: FLARE server URL.
- ``allowSelfSignedCerts``: Allow self-signed certificates.
- ``inFilters``: Optional input filters for data processing.
- ``outFilters``: Optional output filters for result processing.
- ``resolverRegistry``: Optional component resolver registry.

**Methods**

.. code-block:: swift

   // Start federated learning
   func run() async
   
   // Stop federated learning
   func stop()
   
   // Get current status
   var status: NVFlareStatus { get }

Data Sources
============

Implementing Data Sources
-------------------------

Both platforms require implementing a data source interface to provide training data.

**Android DataSource Interface**

.. code-block:: kotlin

   interface DataSource {
       fun getDataset(jobName: String, context: Context): Dataset
   }

**iOS NVFlareDataSource Protocol**

.. code-block:: swift

   protocol NVFlareDataSource {
       func getDataset(for jobName: String, context: NVFlareContext) throws -> NVFlareDataset
   }

**Example Implementation**

.. code-block:: kotlin

   class MyDataSource : DataSource {
       override fun getDataset(jobName: String, context: Context): Dataset {
           return MyDataset()
       }
   }

.. code-block:: swift

   class MyDataSource: NVFlareDataSource {
       func getDataset(for jobName: String, context: NVFlareContext) throws -> NVFlareDataset {
           return MyDataset()
       }
   }

Model Development
=================

ExecuTorch Integration
----------------------

Mobile FL training uses ExecuTorch for optimized model execution. Models must be converted from PyTorch to ExecuTorch format.

**Model Conversion**

.. code-block:: python

   import torch
   from executorch.exir import to_edge_transform_and_lower
   
   # Load your PyTorch model
   model = YourPyTorchModel()
   model.eval()
   
   # Prepare example input
   example_input = torch.randn(1, 3, 224, 224)
   
   # Export the model using torch.export
   exported_program = torch.export.export(model, (example_input,))
   
   # Convert to ExecuTorch format using public API
   edge_program = to_edge_transform_and_lower(exported_program)

**Model Requirements**

- Models must be compatible with ExecuTorch's supported operations.
- Input/output shapes must be fixed at conversion time.
- Custom operations may require ExecuTorch extensions.
- Use the official ExecuTorch export APIs for model conversion.

Best Practices
==============

Performance Optimization
------------------------

1. **Model Size**: Keep models lightweight for mobile constraints.
2. **Batch Size**: Use appropriate batch sizes for device memory.
3. **Training Frequency**: Balance training frequency with battery life.
4. **Data Caching**: Cache frequently used data locally.

Error Handling
--------------

1. **Network Errors**: Implement retry logic for network failures.
2. **Model Errors**: Handle model loading and training errors gracefully.
3. **Data Errors**: Validate data before training.
4. **Timeout Handling**: Implement appropriate timeouts.

Security Considerations
-----------------------

1. **Certificate Validation**: Use proper certificate validation in production.
2. **Data Privacy**: Ensure sensitive data is handled securely.
3. **Model Protection**: Consider model encryption for sensitive applications.
4. **Network Security**: Use HTTPS for all server communication.

Troubleshooting
===============

Common Issues
-------------

**Build Errors**
* Ensure all dependencies are properly linked.
* Check ExecuTorch library compatibility.
* Verify SDK files are correctly copied.

**Runtime Errors**
* Check network connectivity.
* Verify server configuration.
* Review device logs for specific error messages.

**Performance Issues**
* Monitor memory usage during training.
* Optimize model architecture.
* Adjust batch sizes and training parameters.

Examples and Tutorials
======================

Complete working examples are available in the NVIDIA FLARE repository:

* **iOS Example App**: `iOS Example Project <https://github.com/NVIDIA/NVFlare/tree/main/nvflare/edge/device/ios/ExampleProject>`_
* **Android Example App**: `Android Example Project <https://github.com/NVIDIA/NVFlare/tree/main/nvflare/edge/device/android>`_
* **How to Run NVIDIA FLARE with Edge**: `Edge Examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_ - includes both simulation and real devices

.. tip::
   Start with the examples to understand the complete integration flow before building your own application.

Getting Help
============

* **Documentation**: Refer to the main :ref:`FLARE documentation <user_guide>`.
* **Examples**: Check the examples in ``examples/advanced/edge/``.
* **Issues**: Report issues on the `NVIDIA FLARE GitHub repository <https://github.com/NVIDIA/NVFlare>`_.
* **Community**: Join the NVIDIA FLARE community discussions.
* **ExecuTorch Support**: `ExecuTorch documentation <https://pytorch.org/executorch/>`_ for mobile-specific issues.

