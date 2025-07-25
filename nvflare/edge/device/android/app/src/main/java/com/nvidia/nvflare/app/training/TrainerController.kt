package com.nvidia.nvflare.training

import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nvidia.nvflare.sdk.network.Connection
import com.nvidia.nvflare.models.Job
import com.nvidia.nvflare.sdk.network.NVFlareError
import com.nvidia.nvflare.sdk.network.TaskResponse
import com.nvidia.nvflare.sdk.training.TrainingConfig
import com.nvidia.nvflare.sdk.utils.MethodType
import com.nvidia.nvflare.sdk.utils.TrainerType
import com.nvidia.nvflare.sdk.utils.TrainingStatus
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay

class TrainerController(private val connection: Connection) : ViewModel() {
    private val TAG = "TrainerController"
    private val _status = MutableLiveData<TrainingStatus>(TrainingStatus.IDLE)
    val status: LiveData<TrainingStatus> = _status

    private val _trainerType = MutableLiveData<TrainerType>(TrainerType.EXECUTORCH)
    val trainerType: LiveData<TrainerType> = _trainerType

    private val _supportedMethods = MutableLiveData<Set<MethodType>>(setOf(MethodType.CNN, MethodType.XOR))
    val supportedMethods: LiveData<Set<MethodType>> = _supportedMethods

    private var currentTask: kotlinx.coroutines.Job? = null
    private var currentJob: Job? = null

    val capabilities: Map<String, Any>
        get() {
            val methods = _supportedMethods.value?.map { it.displayName } ?: emptyList()
            return mapOf("methods" to methods)
        }

    init {
        // Set initial capabilities
        connection.setCapabilities(capabilities)
    }

    fun toggleMethod(method: MethodType) {
        val currentMethods = _supportedMethods.value ?: emptySet()
        _supportedMethods.value = if (currentMethods.contains(method)) {
            currentMethods - method
        } else {
            currentMethods + method
        }
        connection.setCapabilities(capabilities)
    }

    fun setTrainerType(type: TrainerType) {
        _trainerType.value = type
    }

    fun startTraining() {
        if (_status.value == TrainingStatus.TRAINING) {
            Log.w(TAG, "Training already in progress")
            return
        }

        _status.value = TrainingStatus.TRAINING
        currentTask = viewModelScope.launch {
            try {
                runTrainingLoop()
            } catch (e: Exception) {
                Log.e(TAG, "Training failed", e)
                if (_status.value != TrainingStatus.STOPPING) {
                    _status.value = TrainingStatus.IDLE
                }
                throw e
            }
        }
    }

    fun stopTraining() {
        _status.value = TrainingStatus.STOPPING
        currentTask?.cancel()
        currentTask = null
        _status.value = TrainingStatus.IDLE
        connection.resetCookie()
    }

    private suspend fun runTrainingLoop() {
        var currentJob: Job? = null
        
        // Job fetching loop
        while (currentJob == null && currentTask?.isCancelled != true) {
            try {
                Log.d(TAG, "Fetching job...")
                val jobResponse = connection.fetchJob()
                Log.d(TAG, "Job response: $jobResponse")

                if (jobResponse.status == "stopped") {
                    Log.d(TAG, "Server requested stop")
                    throw NVFlareError.ServerRequestedStop
                }

                val job = jobResponse.toJob()
                val methodString = jobResponse.method ?: ""
                val method = if (methodString.isNotEmpty()) MethodType.fromString(methodString) else MethodType.CNN
                
                // Only skip if a method is specified but not supported
                if (methodString.isNotEmpty() && !(_supportedMethods.value?.contains(method) ?: false)) {
                    Log.d(TAG, "Skipping job with unsupported method: $methodString")
                    delay(5000) // Add delay before retry
                    continue
                }

                currentJob = job
                Log.d(TAG, "Starting job: ${currentJob.id}")
            } catch (e: Exception) {
                Log.d(TAG, "Failed to fetch job, retrying in 5 seconds...", e)
                delay(5000)
                continue
            }
        }

        val job = currentJob ?: run {
            Log.e(TAG, "No valid job found")
            throw NVFlareError.JobFetchFailed("Job fetch failed")
        }

        // Task execution loop
        while (job.status == "running" && currentTask?.isCancelled != true) {
            try {
                Log.d(TAG, "Fetching task for job: ${job.id}")
                val taskResponse = connection.fetchTask(job.id)
                Log.d(TAG, "Task response: $taskResponse")

                // Don't exit on model version 0 or no tasks, just retry
                if (!taskResponse.taskStatus.shouldContinueTraining) {
                    Log.d(TAG, "No tasks available or model not ready, retrying in 5 seconds...")
                    delay(5000)
                    continue
                }

                Log.d(TAG, "Creating trainer for task: ${taskResponse.taskId}")
                val task = taskResponse.toTrainingTask(job.id)
                Log.d(TAG, "Creating trainer for task: ${task.id}")
                
                val trainer = createTrainer(task.modelData, task.trainingConfig)
                Log.d(TAG, "Trainer created successfully")

                Log.d(TAG, "Starting training for task: ${task.id}")
                val weightDiff = trainer.train(task.trainingConfig)
                Log.d(TAG, "Training completed for task: ${task.id}")

                Log.d(TAG, "Sending results for task: ${task.id}")
                connection.sendResult(
                    jobId = job.id,
                    taskId = task.id,
                    taskName = task.name,
                    weightDiff = weightDiff
                )
                Log.d(TAG, "Results sent successfully for task: ${task.id}")

            } catch (e: Exception) {
                Log.e(TAG, "Task execution failed, retrying in 5 seconds...", e)
                if (_status.value != TrainingStatus.STOPPING) {
                    _status.value = TrainingStatus.IDLE
                }
                delay(5000)
                continue
            }
        }
    }

    private fun createTrainer(modelData: String, meta: TrainingConfig): Trainer {
        val methodString = meta.method ?: ""
        val method = MethodType.fromString(methodString)
        
        if (method == null) {
            Log.e(TAG, "Missing or invalid method in job metadata")
            throw NVFlareError.InvalidMetadata("Missing or invalid method in job metadata")
        }

        if (!_supportedMethods.value!!.contains(method)) {
            Log.e(TAG, "Method $methodString is not supported by this client")
            throw NVFlareError.InvalidMetadata("Method $methodString is not supported by this client")
        }

        return when (_trainerType.value) {
            TrainerType.EXECUTORCH -> {
                Log.d(TAG, "Creating ETTrainerWrapper")
                ETTrainerWrapper(modelData, meta)
            }
            else -> throw NVFlareError.InvalidMetadata("Unsupported trainer type")
        }
    }
} 