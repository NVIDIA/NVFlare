package com.nvidia.nvflare.app.controllers

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.app.data.AndroidDataSource
import com.nvidia.nvflare.app.data.DatasetError
import com.nvidia.nvflare.sdk.models.TrainingStatus
import com.nvidia.nvflare.sdk.core.AndroidFlareRunner
import com.nvidia.nvflare.sdk.core.Context as FlareContext
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.Dataset
import com.nvidia.nvflare.sdk.core.Connection
import com.nvidia.nvflare.sdk.trainers.ETTrainerFactory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext


sealed class TrainingError : Exception() {
    object DATASET_CREATION_FAILED : TrainingError()
    object CONNECTION_FAILED : TrainingError()
    object TRAINING_FAILED : TrainingError()
    object NO_SUPPORTED_JOBS : TrainingError()
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
            "supported_jobs" to supportedJobs.map { it.value },
            "methods" to listOf("cnn", "xor")
        )
    
    fun toggleJob(job: SupportedJob) {
        if (supportedJobs.contains(job)) {
            supportedJobs = supportedJobs - job
        } else {
            supportedJobs = supportedJobs + job
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
        
        // Create connection on main thread to avoid background thread issues
        val connection = createConnection()
        
        currentJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "FlareRunnerController: Starting federated learning")
                
                // Create dataset based on supported jobs
                val dataset: Dataset
                val dataSource = AndroidDataSource(context)
                var jobName: String = "federated_learning"
                
                if (supportedJobs.contains(SupportedJob.CIFAR10)) {
                    try {
                        val context = FlareContext()
                        context.put("dataset_name", "cifar10")
                        dataset = dataSource.getDataset("train", context)
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
                    val context = FlareContext()
                    context.put("dataset_name", "xor")
                    dataset = dataSource.getDataset("train", context)
                    jobName = "xor_et"
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
                    connection = connection,
                    jobName = jobName,
                    dataSource = dataSource,
                    deviceInfo = mapOf(
                        "device_id" to (android.provider.Settings.Secure.getString(
                            context.contentResolver, 
                            android.provider.Settings.Secure.ANDROID_ID
                        ) ?: "unknown"),
                        "platform" to "android",
                        "app_version" to context.packageManager.getPackageInfo(context.packageName, 0).versionName
                    ),
                    userInfo = emptyMap(),
                    jobTimeout = 86400.0f  // 24 hours
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
        Log.d(TAG, "FlareRunnerController: Stopping training and cleaning up resources")
        
        // Stop the flare runner
        flareRunner?.stop()
        flareRunner = null
        
        // Cancel the current job
        currentJob?.cancel()
        currentJob = null
        
        // Clear the dataset reference
        currentDataset = null
        
        // Reset status to IDLE so training can be started again
        status = TrainingStatus.IDLE
        
        Log.d(TAG, "FlareRunnerController: Training stopped and resources cleaned up")
    }
    
    private fun createConnection(): com.nvidia.nvflare.sdk.core.Connection {
        val connection = com.nvidia.nvflare.sdk.core.Connection(context)
        connection.hostname.value = serverHost
        connection.port.value = serverPort
        connection.setCapabilities(capabilities)
        return connection
    }
} 