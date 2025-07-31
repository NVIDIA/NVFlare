package com.nvidia.nvflare.app.controllers

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.app.data.AndroidDataSource
import com.nvidia.nvflare.app.data.DatasetError
import com.nvidia.nvflare.sdk.AndroidFlareRunner
import com.nvidia.nvflare.sdk.defs.Context as FlareContext
import com.nvidia.nvflare.sdk.defs.DataSource
import com.nvidia.nvflare.sdk.defs.Signal
import com.nvidia.nvflare.sdk.defs.Dataset
import com.nvidia.nvflare.sdk.trainers.ETTrainerFactory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class TrainingStatus {
    IDLE,
    TRAINING,
    STOPPING
}

enum class TrainingError : Exception {
    DATASET_CREATION_FAILED,
    CONNECTION_FAILED,
    TRAINING_FAILED,
    NO_SUPPORTED_JOBS
}

enum class SupportedJob(val value: String) {
    CIFAR10("CIFAR10"),
    XOR("XOR");
    
    val displayName: String
        get() = when (this) {
            CIFAR10 -> "CIFAR-10"
            XOR -> "XOR"
        }
    
    val datasetType: String
        get() = when (this) {
            CIFAR10 -> "cifar10"
            XOR -> "xor"
        }
}

/**
 * Android app-level coordinator that uses NVFlareSDK.
 * Mirrors the iOS TrainerController pattern with strong reference management.
 */
class FlareRunnerController(
    private val context: Context
) {
    private val TAG = "FlareRunnerController"
    
    // State management
    private var status: TrainingStatus = TrainingStatus.IDLE
    var supportedJobs: Set<SupportedJob> = setOf(SupportedJob.CIFAR10, SupportedJob.XOR)
        private set
    
    // Task management
    private var currentJob: Job? = null
    private var flareRunner: AndroidFlareRunner? = null
    
    // CRITICAL: Strong reference to keep dataset alive during training
    // This prevents the dataset from being deallocated while training is in progress
    private var currentDataset: Dataset? = null
    
    // Server configuration
    var serverHost: String = "192.168.6.101"
    var serverPort: Int = 4321
    
    val capabilities: Map<String, Any>
        get() = mapOf(
            "supported_jobs" to supportedJobs.map { it.value }
        )
    
    fun toggleJob(job: SupportedJob) {
        if (supportedJobs.contains(job)) {
            supportedJobs = supportedJobs - job
        } else {
            supportedJobs = supportedJobs + job
        }
        
        // Update the runner if it exists
        flareRunner?.let { runner ->
            // Note: Android doesn't have updateSupportedJobs method yet, 
            // but we can recreate the runner if needed
        }
    }
    
    fun startTraining(
        onStatusUpdate: (TrainingStatus) -> Unit,
        onError: (Exception) -> Unit,
        onSuccess: () -> Unit
    ) {
        if (status != TrainingStatus.IDLE) return
        
        // Check if any jobs are supported
        if (supportedJobs.isEmpty()) {
            onError(TrainingError.NO_SUPPORTED_JOBS)
            return
        }
        
        status = TrainingStatus.TRAINING
        onStatusUpdate(status)
        
        currentJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "FlareRunnerController: Starting federated learning")
                
                // Create dataset based on supported jobs
                val dataset: Dataset
                val dataSource = AndroidDataSource(context)
                
                if (supportedJobs.contains(SupportedJob.CIFAR10)) {
                    try {
                        dataset = dataSource.getDataset("cifar10", FlareContext())
                        Log.d(TAG, "FlareRunnerController: Created CIFAR-10 dataset")
                        
                        // Validate dataset using SDK's standardized validation
                        dataset.validate()
                        Log.d(TAG, "FlareRunnerController: CIFAR-10 dataset validation passed")
                        Log.d(TAG, "FlareRunnerController: CIFAR-10 dataset size: ${dataset.size()}")
                        
                    } catch (e: DatasetError.NoDataFound) {
                        Log.e(TAG, "FlareRunnerController: CIFAR-10 data not found in app bundle")
                        throw TrainingError.DATASET_CREATION_FAILED
                    } catch (e: DatasetError.InvalidDataFormat) {
                        Log.e(TAG, "FlareRunnerController: CIFAR-10 data format is invalid")
                        throw TrainingError.DATASET_CREATION_FAILED
                    } catch (e: DatasetError.EmptyDataset) {
                        Log.e(TAG, "FlareRunnerController: CIFAR-10 dataset is empty")
                        throw TrainingError.DATASET_CREATION_FAILED
                    } catch (e: Exception) {
                        Log.e(TAG, "FlareRunnerController: Failed to create CIFAR-10 dataset: $e")
                        throw TrainingError.DATASET_CREATION_FAILED
                    }
                } else if (supportedJobs.contains(SupportedJob.XOR)) {
                    dataset = dataSource.getDataset("xor", FlareContext())
                    Log.d(TAG, "FlareRunnerController: Created XOR dataset")
                    Log.d(TAG, "FlareRunnerController: XOR dataset size: ${dataset.size()}")
                } else {
                    throw TrainingError.DATASET_CREATION_FAILED
                }
                
                // Store the dataset to keep it alive during training
                currentDataset = dataset
                Log.d(TAG, "FlareRunnerController: Stored dataset reference: $dataset")
                
                // Create FlareRunner with dataset
                val runner = AndroidFlareRunner(
                    context = context,
                    connection = createConnection(),
                    jobName = "federated_learning",
                    dataSource = dataSource,
                    deviceInfo = mapOf(
                        "device_id" to android.provider.Settings.Secure.getString(
                            context.contentResolver, 
                            android.provider.Settings.Secure.ANDROID_ID
                        ) ?: "unknown",
                        "platform" to "android",
                        "app_version" to context.packageManager.getPackageInfo(context.packageName, 0).versionName
                    ),
                    userInfo = emptyMap(),
                    jobTimeout = 30.0f
                )
                
                flareRunner = runner
                
                Log.d(TAG, "FlareRunnerController: Supported jobs: ${supportedJobs.map { it.value }}")
                Log.d(TAG, "FlareRunnerController: Using app's dataset implementation")
                
                // Start the runner (this will call the main FL loop)
                Log.d(TAG, "FlareRunnerController: About to start training with dataset reference: $currentDataset")
                runner.run()
                
                Log.d(TAG, "FlareRunnerController: Training completed, about to cleanup")
                
                // Reset status after successful completion
                withContext(Dispatchers.Main) {
                    Log.d(TAG, "FlareRunnerController: Clearing dataset reference")
                    currentDataset = null // Release dataset reference
                    status = TrainingStatus.IDLE
                    onStatusUpdate(status)
                    onSuccess()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "FlareRunnerController: Training failed with error: $e")
                withContext(Dispatchers.Main) {
                    Log.d(TAG, "FlareRunnerController: Clearing dataset reference due to error")
                    currentDataset = null // Release dataset reference on error too
                    status = TrainingStatus.IDLE
                    onStatusUpdate(status)
                    onError(e)
                }
            }
        }
    }
    
    fun stopTraining() {
        status = TrainingStatus.STOPPING
        flareRunner?.stop()
        currentJob?.cancel()
    }
    
    private fun createConnection(): com.nvidia.nvflare.sdk.network.Connection {
        return com.nvidia.nvflare.sdk.network.Connection(
            hostname = serverHost,
            port = serverPort,
            deviceInfo = mapOf(
                "device_id" to android.provider.Settings.Secure.getString(
                    context.contentResolver, 
                    android.provider.Settings.Secure.ANDROID_ID
                ) ?: "unknown",
                "platform" to "android"
            )
        )
    }
} 