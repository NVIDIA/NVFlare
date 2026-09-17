package com.nvidia.nvflare.app.controllers

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.app.data.AndroidDataSource
import com.nvidia.nvflare.app.data.DatasetError
import com.nvidia.nvflare.sdk.models.TrainingProgress
import com.nvidia.nvflare.sdk.training.TrainingStatus
import com.nvidia.nvflare.sdk.core.AndroidFlareRunner
import com.nvidia.nvflare.sdk.core.Connection
import com.nvidia.nvflare.sdk.core.Context as FlareContext
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.Dataset


import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import android.os.Handler
import android.os.Looper


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
    companion object {
        // Timeout constants
        private const val TWENTY_FOUR_HOURS_IN_SECONDS = 86400.0f
    }
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
    var useHttps: Boolean = false
    var allowSelfSignedCerts: Boolean = false
    
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
        onProgressUpdate: (TrainingProgress) -> Unit,
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
        onProgressUpdate(TrainingProgress.idle())
        
        // Create connection on main thread to avoid background thread issues
        val connection = createConnection()
        
        currentJob = CoroutineScope(Dispatchers.IO).launch {
            try {
                Log.d(TAG, "FlareRunnerController: Starting federated learning")
                
                // First: Determine which job to use based on user selection
                val selectedJob = when {
                    supportedJobs.contains(SupportedJob.CIFAR10) && supportedJobs.contains(SupportedJob.XOR) -> {
                        // If both are enabled, prefer CIFAR-10 (can be changed to user preference)
                        Log.d(TAG, "FlareRunnerController: Both jobs enabled, selecting CIFAR-10")
                        SupportedJob.CIFAR10
                    }
                    supportedJobs.contains(SupportedJob.CIFAR10) -> {
                        Log.d(TAG, "FlareRunnerController: CIFAR-10 job selected")
                        SupportedJob.CIFAR10
                    }
                    supportedJobs.contains(SupportedJob.XOR) -> {
                        Log.d(TAG, "FlareRunnerController: XOR job selected")
                        SupportedJob.XOR
                    }
                    else -> {
                        Log.e(TAG, "FlareRunnerController: No supported jobs enabled. Current supported jobs: ${supportedJobs.map { it.value }}")
                        throw TrainingError.NO_SUPPORTED_JOBS
                    }
                }
                
                // Second: Create dataset based on selected job
                val dataSource = AndroidDataSource(context)
                val dataset: Dataset
                val jobName: String
                
                when (selectedJob) {
                    SupportedJob.CIFAR10 -> {
                        jobName = "cifar10_et"
                        try {
                            dataset = dataSource.getDataset(jobName, FlareContext())
                            Log.d(TAG, "FlareRunnerController: Created CIFAR-10 dataset for job: $jobName")
                            
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
                    }
                    SupportedJob.XOR -> {
                        jobName = "xor_et"
                        dataset = dataSource.getDataset(jobName, FlareContext())
                        Log.d(TAG, "FlareRunnerController: Created XOR dataset for job: $jobName")
                        Log.d(TAG, "FlareRunnerController: XOR dataset size: ${dataset.size()}")
                    }
                }
                
                Log.d(TAG, "FlareRunnerController: Selected job: ${selectedJob.displayName} -> $jobName")
                
                // Store the dataset to keep it alive during training
                currentDataset = dataset
                Log.d(TAG, "FlareRunnerController: Stored dataset reference: $dataset")
                
                                // Create FlareRunner with dataset and progress callback
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
                    jobTimeout = TWENTY_FOUR_HOURS_IN_SECONDS,
                    onProgressUpdate = { progress ->
                        // Pass TrainingProgress directly to UI on main thread
                        Handler(Looper.getMainLooper()).post {
                            onProgressUpdate(progress)
                        }
                    }
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
        
        // Configure SSL settings
        if (useHttps) {
            connection.setScheme("https")
        }
        connection.setAllowSelfSignedCerts(allowSelfSignedCerts)
        
        // Set capabilities
        connection.setCapabilities(capabilities)
        
        // Set user info - use device ID as user ID for now, can be made configurable later
        val userId = android.provider.Settings.Secure.getString(
            context.contentResolver, 
            android.provider.Settings.Secure.ANDROID_ID
        ) ?: "unknown_user"
        connection.setUserInfo(mapOf("user_id" to userId))
        
        return connection
    }
}
