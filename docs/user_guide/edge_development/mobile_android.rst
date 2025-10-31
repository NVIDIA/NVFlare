.. _mobile_android_api:

#########################
Android SDK API Reference
#########################

This document provides a comprehensive API reference for the NVIDIA FLARE Android SDK, enabling federated learning on Android devices using ExecuTorch.

.. note::
   This API reference assumes familiarity with the :ref:`mobile development guide <flare_mobile>` and basic Android development concepts.

Overview
========

The Android SDK provides native Kotlin/Java libraries for implementing federated learning on Android devices. The SDK handles communication with FLARE servers, model training using ExecuTorch, and data management.

Key Components
==============

* **AndroidFlareRunner**: Main orchestrator for federated learning.
* **Connection**: HTTP/HTTPS communication with FLARE servers.
* **ETTrainer**: ExecuTorch-based model training.
* **DataSource**: Interface for providing training data.
* **Dataset**: Data interface for training examples.

AndroidFlareRunner
==================

The main orchestrator for federated learning on Android devices. Handles job fetching, task execution, result reporting, component resolution, filtering, and event handling.

Constructor
-----------

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

Parameters
~~~~~~~~~~

* ``context``: Android application context.
* ``connection``: Connection instance for server communication.
* ``jobName``: Name of the FL job to participate in.
* ``dataSource``: Data source providing training data.
* ``deviceInfo``: Device metadata (``device_id``, ``platform``, etc.).
* ``userInfo``: User metadata (``user_id``, etc.).
* ``jobTimeout``: Timeout in seconds for job operations.
* ``inFilters``: Optional input filters for data processing.
* ``outFilters``: Optional output filters for result processing.
* ``resolverRegistry``: Optional component resolver registry.

What is a Resolver?
-------------------

A **Resolver** is a component that maps string identifiers to actual class implementations. In the context of FLARE's edge SDK, resolvers are used to dynamically instantiate training components, filters, and other plugins based on configuration data received from the server.

For example, when the server sends a job configuration that specifies a trainer component, the resolver looks up the string identifier (like "ETTrainerExecutor") and maps it to the actual class that should be instantiated. This allows for flexible, configuration-driven component loading without hardcoding specific implementations.

The ``resolverRegistry`` parameter allows you to register custom resolvers for your own components, enabling the system to dynamically load and instantiate them as needed.

Properties
----------

.. code-block:: kotlin

   val jobName: String
   // The name of the federated learning job

Methods
-------

run()
~~~~~~

Starts the main federated learning loop. This method runs continuously until the job is complete or stopped.

.. code-block:: kotlin

   fun run()

**Usage:**

.. code-block:: kotlin

   lifecycleScope.launch {
       flareRunner.run()
   }

stop()
~~~~~~

Stops the federated learning process and cleans up resources.

.. code-block:: kotlin

   fun stop()

**Usage:**

.. code-block:: kotlin

   override fun onDestroy() {
       super.onDestroy()
       flareRunner.stop()
   }

Built-in Component Resolvers
----------------------------

The ``AndroidFlareRunner`` includes built-in resolvers for common components:

* ``Executor.ETTrainerExecutor``: ExecuTorch-based training executor.
* ``Trainer.DLTrainer``: Deep learning trainer (mapped to ``ETTrainerExecutor``).
* ``Filter.NoOpFilter``: No-operation filter.
* ``EventHandler.NoOpEventHandler``: No-operation event handler.
* ``Batch.SimpleBatch``: Simple batch processing.

Connection
==========

Manages HTTP/HTTPS communication with FLARE servers. Handles authentication, certificate validation, and request/response processing.

Constructor
-----------

.. code-block:: kotlin

   Connection(context: Context)

Parameters
~~~~~~~~~~

* ``context``: Android application context

Properties
----------

.. code-block:: kotlin

   val hostname: MutableLiveData<String>
   // Server hostname (observable)

   val port: MutableLiveData<Int>
   // Server port (observable)

   val isValid: Boolean
   // Whether the connection configuration is valid

   fun getUserInfo(): Map<String, String>
   // Get current user information

Methods
-------

setCapabilities(capabilities)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sets device capabilities for the connection.

.. code-block:: kotlin

   fun setCapabilities(capabilities: Map<String, Any>)

**Parameters:**
* ``capabilities``: Map of device capabilities.

setUserInfo(userInfo)
~~~~~~~~~~~~~~~~~~~~~

Sets user information for the connection.

.. code-block:: kotlin

   fun setUserInfo(userInfo: Map<String, String>)

**Parameters:**
* ``userInfo``: Map of user information.

setScheme(scheme)
~~~~~~~~~~~~~~~~~

Sets the HTTP scheme (http/https).

.. code-block:: kotlin

   fun setScheme(scheme: String)

**Parameters:**
* ``scheme``: ``"http"`` or ``"https"``.

setAllowSelfSignedCerts(allow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configures whether to allow self-signed certificates.

.. code-block:: kotlin

   fun setAllowSelfSignedCerts(allow: Boolean)

**Parameters:**
* ``allow``: ``true`` to allow self-signed certificates.

.. warning::
   Allowing self-signed certificates creates security vulnerabilities. Only use in development or controlled environments.

getJob(jobName, deviceInfo, userInfo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requests a job from the server.

.. code-block:: kotlin

   suspend fun getJob(
       jobName: String,
       deviceInfo: Map<String, String>,
       userInfo: Map<String, String>
   ): JobResponse?

**Parameters:**
* ``jobName``: Name of the job to request.
* ``deviceInfo``: Device information.
* ``userInfo``: User information.

**Returns:** ``JobResponse`` if successful, ``null`` otherwise.

getTask(jobId, taskName)
~~~~~~~~~~~~~~~~~~~~~~~~

Requests a task from the server.

.. code-block:: kotlin

   suspend fun getTask(
       jobId: String,
       taskName: String
   ): TaskResponse?

**Parameters:**
* ``jobId``: Job identifier.
* ``taskName``: Name of the task to request.

**Returns:** ``TaskResponse`` if successful, ``null`` otherwise.

reportResult(jobId, taskId, result)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reports task results to the server.

.. code-block:: kotlin

   suspend fun reportResult(
       jobId: String,
       taskId: String,
       result: Map<String, Any>
   ): ResultResponse?

**Parameters:**
* ``jobId``: Job identifier.
* ``taskId``: Task identifier.
* ``result``: Task execution results.

**Returns:** ``ResultResponse`` if successful, ``null`` otherwise.

ETTrainer
=========

ExecuTorch-based trainer for on-device model training. Implements ``AutoCloseable`` for proper resource management.

Constructor
-----------

.. code-block:: kotlin

   ETTrainer(
       context: android.content.Context,
       meta: Map<String, Any>,
       dataset: Dataset? = null
   )

Parameters
~~~~~~~~~~

* ``context``: Android application context.
* ``meta``: Model metadata.
* ``dataset``: Optional dataset for training.

Methods
-------

train(config, dataset, modelData)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trains the model using the provided configuration and dataset.

.. code-block:: kotlin

   @Throws(Exception::class)
   fun train(
       config: TrainingConfig,
       dataset: Dataset,
       modelData: ByteArray
   ): Map<String, Any>

**Parameters:**
* ``config``: Training configuration.
* ``dataset``: Training dataset.
* ``modelData``: Model data in ExecuTorch format.

**Returns:** Training results including loss and predictions.

**Throws:** ``Exception`` if training fails.

**Usage:**

.. code-block:: kotlin

   ETTrainer(context, meta, dataset).use { trainer ->
       val result = trainer.train(config, dataset, modelData)
   }

close()
~~~~~~~

Closes the trainer and releases resources.

.. code-block:: kotlin

   override fun close()

DataSource Interface
====================

Interface for providing training data to the FL system.

Interface Definition
--------------------

.. code-block:: kotlin

   interface DataSource {
       fun getDataset(jobName: String, context: Context): Dataset
   }

Methods
-------

getDataset(jobName, context)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Retrieves a dataset for the specified job.

.. code-block:: kotlin

   fun getDataset(jobName: String, context: Context): Dataset

**Parameters:**
* ``jobName``: Name of the federated learning job.
* ``context``: FLARE context.

**Returns:** ``Dataset`` instance for training.

**Example Implementation:**

.. code-block:: kotlin

   class MyDataSource : DataSource {
       override fun getDataset(jobName: String, context: Context): Dataset {
           return when (jobName) {
               "cifar10_job" -> CIFAR10Dataset(context)
               "xor_job" -> XORDataset("train")
               else -> throw IllegalArgumentException("Unknown job: $jobName")
           }
       }
   }

Dataset Interface
=================

Interface for providing training examples to the trainer.

Interface Definition
--------------------

.. code-block:: kotlin

   interface Dataset {
       fun size(): Int
       fun getBatch(batchSize: Int): List<Map<String, Any>>
   }

Methods
-------

size()
~~~~~~

Returns the total number of examples in the dataset.

.. code-block:: kotlin

   fun size(): Int

**Returns:** Number of examples.

getBatch(batchSize)
~~~~~~~~~~~~~~~~~~~

Retrieves a batch of training examples.

.. code-block:: kotlin

   fun getBatch(batchSize: Int): List<Map<String, Any>>

**Parameters:**
* ``batchSize``: Number of examples to return.

**Returns:** List of training examples.

**Example Implementation:**

.. code-block:: kotlin

   class MyDataset : Dataset {
       private val data = mutableListOf<Map<String, Any>>()
       
       override fun size(): Int = data.size
       
       override fun getBatch(batchSize: Int): List<Map<String, Any>> {
           return data.shuffled().take(batchSize)
       }
   }

TrainingConfig
==============

Configuration class for training parameters.

Properties
----------

.. code-block:: kotlin

   val localEpochs: Int
   // Number of local training epochs

   val localBatchSize: Int
   // Batch size for local training

   val localLearningRate: Float
   // Learning rate for local training

   val localMomentum: Float
   // Momentum for local training

   val inFilters: List<Filter>?
   // Input filters

   val outFilters: List<Filter>?
   // Output filters

Usage Examples
==============

Basic Setup
-----------

.. code-block:: kotlin

   class MainActivity : AppCompatActivity() {
       private lateinit var flareRunner: AndroidFlareRunner
       
       override fun onCreate(savedInstanceState: Bundle?) {
           super.onCreate(savedInstanceState)
           
           // Create connection
           val connection = Connection(this)
           connection.setScheme("https")
           connection.setAllowSelfSignedCerts(false) // Use true for development only
           
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

Custom Data Source
------------------

.. code-block:: kotlin

   class CIFAR10DataSource : DataSource {
       override fun getDataset(jobName: String, context: Context): Dataset {
           return CIFAR10Dataset(context)
       }
   }

Custom Dataset
--------------

.. code-block:: kotlin

   class XORDataset(private val split: String) : Dataset {
       private val data = generateXORData()
       
       override fun size(): Int = data.size
       
       override fun getBatch(batchSize: Int): List<Map<String, Any>> {
           return data.shuffled().take(batchSize)
       }
       
       private fun generateXORData(): List<Map<String, Any>> {
           // Generate XOR training data
           return listOf(
               mapOf("input" to floatArrayOf(0f, 0f), "label" to 0f),
               mapOf("input" to floatArrayOf(0f, 1f), "label" to 1f),
               mapOf("input" to floatArrayOf(1f, 0f), "label" to 1f),
               mapOf("input" to floatArrayOf(1f, 1f), "label" to 0f)
           )
       }
   }

Error Handling
==============

The Android SDK provides comprehensive error handling through exceptions and logging.

Common Exceptions
-----------------

* ``NVFlareError`` (``com.nvidia.nvflare.sdk.core.NVFlareError``): Custom base exception for FLARE-related errors.
* ``IOException`` (``java.io.IOException``): Standard Java exception for network communication errors.
* ``RuntimeException`` (``java.lang.RuntimeException``): Standard Java exception for general runtime errors.

Exception Hierarchy
-------------------

The SDK uses a custom exception hierarchy where ``NVFlareError`` extends ``Exception`` and provides specific error types. In practice, the Android app primarily handles ``ServerRequestedStop`` specifically, while other errors are handled generically:

.. code-block:: kotlin

   sealed class NVFlareError : Exception() {
       // Network related
       data class JobFetchFailed(override val message: String) : NVFlareError()
       data class TaskFetchFailed(override val message: String) : NVFlareError()
       data class InvalidRequest(override val message: String) : NVFlareError()
       data class AuthError(override val message: String) : NVFlareError()
       data class ServerError(override val message: String) : NVFlareError()
       data class NetworkError(override val message: String) : NVFlareError()
       
       // Training related
       data class InvalidMetadata(override val message: String) : NVFlareError()
       data class InvalidModelData(override val message: String) : NVFlareError()
       data class TrainingFailed(override val message: String) : NVFlareError()
       object ServerRequestedStop : NVFlareError()
   }

Error Handling Best Practices
-----------------------------

The Android SDK uses a simplified error handling approach that catches generic exceptions and provides specific handling for ``NVFlareError.ServerRequestedStop``:

.. code-block:: kotlin

   try {
       val result = flareRunner.run()
   } catch (e: Exception) {
       Log.e("FLARE", "Training failed with error: $e")
       
       // Check for specific NVFlareError types
       if (e is NVFlareError.ServerRequestedStop) {
           Log.i("FLARE", "Server requested stop")
           // Gracefully stop training
       } else {
           // Handle other errors generically
           Log.e("FLARE", "Error: ${e.message}")
       }
   }

.. note::
   The Connection class does use more specific error handling, converting ``IOException`` to ``NVFlareError.NetworkError`` and throwing appropriate ``NVFlareError`` subtypes based on HTTP status codes. However, the main application code uses the simplified approach shown above.

Logging
-------

The SDK uses Android's standard logging system. Enable debug logging to see detailed information:

.. code-block:: kotlin

   if (BuildConfig.DEBUG) {
       Log.d("AndroidFlareRunner", "Starting federated learning")
   }

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

**Certificate Errors**
* Use proper certificate validation in production.
* Consider certificate pinning for enhanced security.
* Test with self-signed certificates in development only.

Best Practices
==============

* **Resource Management**: Always use try-with-resources or ``AutoCloseable`` for ``ETTrainer``.
* **Error Handling**: Implement comprehensive error handling and logging.
* **Security**: Use proper certificate validation in production.
* **Performance**: Monitor memory usage and optimize model size.
* **Testing**: Test with various network conditions and device configurations.

For more information, see the :ref:`mobile development guide <flare_mobile>` and `edge examples <https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/edge>`_
