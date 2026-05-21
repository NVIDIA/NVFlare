package com.nvidia.nvflare.sdk.core

import android.content.Context as AndroidContext
import android.util.Log
import com.nvidia.nvflare.sdk.core.Connection
import com.nvidia.nvflare.sdk.models.JobResponse
import com.nvidia.nvflare.sdk.models.TaskResponse
import com.nvidia.nvflare.sdk.models.ResultResponse
import com.nvidia.nvflare.sdk.models.TrainingProgress
import com.nvidia.nvflare.sdk.core.NVFlareError
import com.nvidia.nvflare.sdk.utils.asMap
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.ContextKey
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Filter
import com.nvidia.nvflare.sdk.utils.TaskHeaderKey
import com.nvidia.nvflare.sdk.core.NoOpFilter
import com.nvidia.nvflare.sdk.core.NoOpEventHandler

import com.nvidia.nvflare.sdk.core.SimpleBatch
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.ETTrainerExecutor
import com.nvidia.nvflare.sdk.training.ETTrainer
import com.nvidia.nvflare.sdk.config.processTrainConfig
import com.nvidia.nvflare.sdk.core.EventType
import com.nvidia.nvflare.sdk.core.Executor

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay
import java.util.HashMap

/**
 * Main orchestrator for federated learning on Android edge devices.
 * Handles job fetching, task execution, result reporting, component resolution,
 * filtering, and event handling. Consolidates all FL functionality into a single class.
 * 
 * @param onProgressUpdate Optional callback to receive training progress updates.
 *                         Receives TrainingProgress objects with detailed status information.
 */
class AndroidFlareRunner(
    private val context: AndroidContext,
    private val connection: Connection,
    val jobName: String,
    private val dataSource: DataSource,
    private val deviceInfo: Map<String, String>,
    private val userInfo: Map<String, String>,
    private val jobTimeout: Float,
    private val inFilters: List<Filter>? = null,
    private val outFilters: List<Filter>? = null,
    private val resolverRegistry: Map<String, Class<*>>? = null,
    private val onProgressUpdate: ((TrainingProgress) -> Unit)? = null
) {
    private val TAG = "AndroidFlareRunner"
    private val abortSignal = Signal()
    private var jobId: String? = null
    private var cookie: Any? = null
    private var currentJobId: String? = null
    private val resolverRegistryMap: MutableMap<String, Class<*>> = HashMap()
    
    // Track training progress
    // Note: These are reset for each job to ensure correct tracking on restart
    private var sessionStartTime: Long = 0
    private var currentPhaseStartTime: Long = 0

    init {
        // Add built-in resolvers
        addBuiltinResolvers()
        
        // Add app-provided resolvers
        resolverRegistry?.let { registry ->
            resolverRegistryMap.putAll(registry)
        }
    }

    /**
     * Add built-in component resolvers.
     */
    private fun addBuiltinResolvers() {
        // Add Android-specific component resolvers here
        // Follow the Python pattern: component_type -> class_reference
        resolverRegistryMap.putAll(mapOf(
            "Executor.ETTrainerExecutor" to ETTrainerExecutor::class.java,
            "Trainer.DLTrainer" to ETTrainerExecutor::class.java,  // Map Trainer.DLTrainer to ETTrainerExecutor
            "Filter.NoOpFilter" to NoOpFilter::class.java,
            "EventHandler.NoOpEventHandler" to NoOpEventHandler::class.java,
            "Batch.SimpleBatch" to SimpleBatch::class.java
        ))
        
        // Note: ETTrainer instances are created directly by ETTrainerExecutorFactory
        // Future training methods can be added there without code changes to this class
        
        // Note: datasource resolver is provided by the app
    }

    /**
     * Get Android context for platform-specific operations.
     */
    private fun getAndroidContext(): AndroidContext {
        return context
    }

    /**
     * Main run loop that continuously processes jobs.
     */
    fun run() {
        Log.d(TAG, "Starting AndroidFlareRunner")
        while (!abortSignal.isTriggered) {
            val sessionDone = doOneJob()
            if (sessionDone) {
                Log.d(TAG, "Session completed")
                break
            }
        }
        Log.d(TAG, "AndroidFlareRunner stopped")
    }

    /**
     * Stop the runner.
     */
    fun stop() {
        Log.d(TAG, "Stopping AndroidFlareRunner")
        abortSignal.trigger(null)
    }

    /**
     * Process one complete job.
     */
    private fun doOneJob(): Boolean {
        sessionStartTime = System.currentTimeMillis()
        val ctx = Context()
        ctx[ContextKey.RUNNER] = this
        ctx[ContextKey.DATA_SOURCE] = dataSource

        // Emit connecting progress with server info
        currentPhaseStartTime = System.currentTimeMillis()
        val serverUrl = "${connection.hostname.value}:${connection.port.value}"
        emitProgress(TrainingProgress.connecting(serverUrl = serverUrl))

        // Get dataset from data source and store in context (iOS pattern)
        val dataset = try {
            dataSource.getDataset(jobName, ctx)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get dataset for job: $jobName", e)
            emitProgress(TrainingProgress.error(
                errorMessage = "Failed to load dataset: ${e.message}",
                errorDetails = e.stackTraceToString()
            ))
            return true  // Cannot continue without dataset
        }
        ctx[ContextKey.DATASET] = dataset
        val datasetSize = dataset.size()
        Log.d(TAG, "Got dataset from data source for job: $jobName, size: $datasetSize")

        // Emit fetching job progress with dataset info
        currentPhaseStartTime = System.currentTimeMillis()
        emitProgress(TrainingProgress.fetchingJob(datasetSize = datasetSize))

        // Try to get a job
        val job = getJob(ctx, abortSignal)
        if (job == null) {
            Log.d(TAG, "No job available")
            emitProgress(TrainingProgress.completed())
            return true
        }

        jobId = job["job_id"] as? String
        val jobData = job["job_data"] as? Map<String, Any>

        Log.d(TAG, "Processing job: $jobName (ID: $jobId)")
        Log.d(TAG, "Job data keys: ${jobData?.keys}")
        
        // Emit job received progress
        val jobFetchDuration = System.currentTimeMillis() - currentPhaseStartTime
        currentPhaseStartTime = System.currentTimeMillis()
        emitProgress(TrainingProgress.jobReceived(
            jobId = jobId ?: "",
            jobName = jobName,
            duration = jobFetchDuration
        ))

        // Process training configuration
        val trainConfig = if (jobData != null) {
            // Extract config from job_data if it exists, otherwise use job_data directly
            val configValue = jobData["config"]
            Log.d(TAG, "Config value type: ${configValue?.javaClass?.simpleName}")
            Log.d(TAG, "Config value: $configValue")
            
            val config = when (configValue) {
                is Map<*, *> -> configValue as Map<String, Any>
                is com.google.gson.JsonObject -> {
                    // Convert JsonObject to Map
                    try {
                        com.google.gson.Gson().fromJson(configValue, Map::class.java) as Map<String, Any>
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to convert JsonObject to Map: $configValue", e)
                        jobData
                    }
                }
                is String -> {
                    // If config is a JSON string, parse it
                    try {
                        com.google.gson.Gson().fromJson(configValue, Map::class.java) as Map<String, Any>
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to parse config JSON: $configValue", e)
                        jobData
                    }
                }
                else -> jobData
            }
            Log.d(TAG, "Job data: $jobData")
            Log.d(TAG, "Extracted config: $config")
            Log.d(TAG, "Config keys: ${config.keys}")
            
            // Add job name to config for method determination
            val configWithJobName = config.toMutableMap()
            configWithJobName[TaskHeaderKey.JOB_NAME] = jobName
            Log.d(TAG, "Config with job name: $configWithJobName")
            
            processTrainConfig(getAndroidContext(), configWithJobName, resolverRegistryMap)
        } else {
            throw RuntimeException("No job data available")
        }

        ctx[ContextKey.COMPONENTS] = trainConfig.objects
        ctx[ContextKey.EVENT_HANDLERS] = trainConfig.eventHandlers ?: emptyList<Any>()

        // Set up filters
        val inputFilters = mutableListOf<Filter>()
        inFilters?.let { inputFilters.addAll(it) }
        trainConfig.inFilters?.let { inputFilters.addAll(it) }

        val outputFilters = mutableListOf<Filter>()
        outFilters?.let { outputFilters.addAll(it) }
        trainConfig.outFilters?.let { outputFilters.addAll(it) }

        // Track rounds for this job (reset for each new job)
        var currentRound = 0
        var totalRounds = 0

        // Process tasks
        while (!abortSignal.isTriggered) {
            val (task, sessionDone) = getTask(ctx, abortSignal)
            
            if (abortSignal.isTriggered) {
                emitProgress(TrainingProgress.stopping())
                return true
            }

            if (task == null) {
                Log.d(TAG, "No more tasks for this job")
                if (sessionDone) {
                    emitProgress(TrainingProgress.completed())
                }
                return sessionDone
            }

            // Create task context
            val taskCtx = Context()
            taskCtx.putAll(ctx)

            // Extract task information
            cookie = task?.get("cookie")
            val taskName = task?.get("task_name") as? String
            val taskData = task?.get("task_data") as? Map<String, Any>

            Log.d(TAG, "Processing task: $taskName")
            
            // Extract round info from task metadata
            val taskMeta = (taskData?.get("meta") as? Map<String, Any>)
            val contributionRound = (taskMeta?.get("contribution_round") as? Number)?.toInt()
            val numRounds = (taskMeta?.get("num_rounds") as? Number)?.toInt()
            
            // Use server's round number
            if (contributionRound != null) {
                currentRound = contributionRound
            } else {
                // Fallback: increment locally if server doesn't provide round number
                currentRound++
            }
            
            if (numRounds != null && numRounds > 0) {
                totalRounds = numRounds
            }
            
            emitProgress(TrainingProgress.fetchingTask(
                currentRound = currentRound,
                totalRounds = totalRounds
            ))
            
            emitProgress(TrainingProgress.taskReceived(
                taskName = taskName ?: "",
                currentRound = currentRound,
                totalRounds = totalRounds
            ))

            // Convert task data to DXO
            val taskDxo = if (taskData != null) {
                DXO.fromMap(taskData)
            } else {
                throw RuntimeException("No task data available")
            }

            // Find executor
            val executor = trainConfig.findExecutor(taskName ?: "")
                ?: throw RuntimeException("Cannot find executor for task $taskName")

            if (executor !is Executor) {
                throw RuntimeException("Bad executor for task $taskName: expected Executor but got ${executor::class.java}")
            }

            // Set task context
            taskCtx[ContextKey.TASK_ID] = task?.get("task_id") ?: ""
            taskCtx[ContextKey.TASK_NAME] = taskName ?: ""
            taskCtx[ContextKey.TASK_DATA] = taskData ?: emptyMap<String, Any>()
            taskCtx[ContextKey.EXECUTOR] = executor
            taskCtx[ContextKey.ANDROID_CONTEXT] = getAndroidContext()

            // Apply input filters
            var filteredTaskDxo = applyFilters(taskDxo, inputFilters, taskCtx)
            if (filteredTaskDxo !is DXO) {
                throw RuntimeException("Task data after filtering is not valid DXO: ${filteredTaskDxo::class.java}")
            }

            if (abortSignal.isTriggered) {
                emitProgress(TrainingProgress.stopping())
                return true
            }

            // Emit training progress
            val trainingStartTime = System.currentTimeMillis()
            emitProgress(TrainingProgress.training(
                taskName = taskName ?: "",
                currentRound = currentRound,
                totalRounds = totalRounds,
                currentEpoch = 0,
                totalEpochs = 1
            ))

            // Execute task
            taskCtx.fireEvent(EventType.BEFORE_TRAIN, System.currentTimeMillis(), abortSignal)
            val output = executor.execute(filteredTaskDxo, taskCtx, abortSignal)
            val trainingDuration = System.currentTimeMillis() - trainingStartTime
            
            Log.d(TAG, "Training completed in ${trainingDuration}ms")

            if (output !is DXO) {
                throw RuntimeException("Output from ${executor::class.java} is not a valid DXO: ${output::class.java}")
            }

            taskCtx.fireEvent(EventType.AFTER_TRAIN, Pair(System.currentTimeMillis(), output), abortSignal)

            if (abortSignal.isTriggered) {
                emitProgress(TrainingProgress.stopping())
                return true
            }

            // Apply output filters
            var filteredOutput = applyFilters(output, outputFilters, taskCtx)
            if (filteredOutput !is DXO) {
                throw RuntimeException("Output after filtering is not valid DXO: ${filteredOutput::class.java}")
            }

            // Emit sending results progress
            val sendStartTime = System.currentTimeMillis()
            emitProgress(TrainingProgress.sendingResults(
                taskName = taskName ?: "",
                currentRound = currentRound,
                totalRounds = totalRounds
            ))

            // Report result
            val result = reportResult(taskCtx, filteredOutput)
            val sendDuration = System.currentTimeMillis() - sendStartTime
            if (!result) {
                Log.e(TAG, "Failed to report result")
                emitProgress(TrainingProgress.error(
                    errorMessage = "Failed to send results to server",
                    errorDetails = "Network error or server rejected the result"
                ))
                return true
            }
            
            // Emit results sent progress
            emitProgress(TrainingProgress.resultsSent(
                currentRound = currentRound,
                totalRounds = totalRounds,
                duration = sendDuration
            ))
        }

        return false
    }

    /**
     * Get a job from the server.
     */
    private fun getJob(ctx: Context, abortSignal: Signal): Map<String, Any>? {
        val startTime = System.currentTimeMillis()
        
        while (true) {
            if (abortSignal.isTriggered) {
                Log.d(TAG, "Job fetch aborted")
                return null
            }
            
            // Check job timeout
            val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0f
            if (elapsedTime > jobTimeout) {
                Log.d(TAG, "Job fetch timed out after ${jobTimeout}s")
                return null
            }
            
            try {
                val jobResponse = runBlocking {
                    connection.fetchJob(jobName)  // Pass job_name to fetchJob
                }

                when (jobResponse.status) {
                    "stopped" -> {
                        Log.d(TAG, "Server requested stop")
                        return null
                    }
                    "OK" -> {
                        currentJobId = jobResponse.jobId
                        
                        // Convert JobResponse to the format expected by FlareRunner
                        return mapOf(
                            "job_id" to (jobResponse.jobId ?: ""),
                            TaskHeaderKey.JOB_NAME to jobName,
                            "job_data" to (jobResponse.jobData?.asMap() ?: emptyMap<String, Any>())
                        )
                    }
                    "RETRY" -> {
                        val retryWait = jobResponse.retryWait ?: 5000L
                        Log.d(TAG, "Server requested retry, waiting ${retryWait}ms")
                        runBlocking { delay(retryWait.toLong()) }
                        continue
                    }
                    else -> {
                        Log.d(TAG, "Job fetch failed with status: ${jobResponse.status}")
                        runBlocking { delay(5000) }
                        continue
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching job", e)
                if (e is NVFlareError.ServerRequestedStop) {
                    return null
                }
                // Retry after delay
                runBlocking { delay(5000L) }
                continue
            }
        }
    }

    /**
     * Get a task from the server.
     */
    private fun getTask(ctx: Context, abortSignal: Signal): Pair<Map<String, Any>?, Boolean> {
        val startTime = System.currentTimeMillis()
        
        while (true) {
            if (abortSignal.isTriggered) {
                Log.d(TAG, "Task fetch aborted")
                return Pair(null, true)
            }
            
            // Check job timeout for task fetching
            val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0f
            if (elapsedTime > jobTimeout) {
                Log.d(TAG, "Task fetch timed out after ${jobTimeout}s")
                return Pair(null, true)
            }
            
            val jobId = currentJobId ?: return Pair(null, true)

            try {
                val taskResponse = runBlocking {
                    connection.fetchTask(jobId)
                }

                // Handle FSM transitions based on task status (matching iOS behavior)
                when {
                    taskResponse.taskStatus.isTerminal -> {
                        // DONE/INVALID/ERROR -> END session
                        Log.d(TAG, "Task session terminated with status: ${taskResponse.taskStatus}")
                        return Pair(null, true)
                    }
                    taskResponse.taskStatus.shouldLookForNewJob -> {
                        // NO_JOB -> go back to getJob (job completed, look for new jobs)
                        Log.d(TAG, "No job available, exiting task loop")
                        return Pair(null, false)
                    }
                    taskResponse.taskStatus.shouldRetryTask -> {
                        // RETRY/NO_TASK -> wait and retry
                        val retryWait = taskResponse.retryWait?.toLong() ?: 5000L
                        Log.d(TAG, "Retry task fetch requested (${taskResponse.taskStatus}), waiting ${retryWait}ms")
                        runBlocking { delay(retryWait) }
                        continue
                    }
                    taskResponse.taskStatus.shouldContinueTraining -> {
                        // OK -> process task
                        // Extract model data properly to avoid corruption
                        val modelData = when (taskResponse.taskData?.data) {
                            is com.google.gson.JsonPrimitive -> {
                                try {
                                    taskResponse.taskData.data.asString
                                } catch (e: Exception) {
                                    taskResponse.taskData.data.toString()
                                }
                            }
                            is com.google.gson.JsonObject -> {
                                taskResponse.taskData.data.toString()
                            }
                            else -> {
                                taskResponse.taskData?.data?.toString() ?: ""
                            }
                        }
                        
                        val taskMap: Map<String, Any> = mapOf(
                            "task_id" to (taskResponse.taskId ?: ""),
                            "task_name" to (taskResponse.taskName ?: ""),
                            "task_data" to mapOf(
                                "data" to mapOf("model" to modelData),
                                "meta" to (taskResponse.taskData?.meta?.asMap() ?: emptyMap<String, Any>()),
                                "kind" to (taskResponse.taskData?.kind ?: "")
                            ),
                            "cookie" to (taskResponse.cookie?.asMap() ?: emptyMap<String, Any>())
                        )
                        return Pair(taskMap, false)
                    }
                    else -> {
                        // UNKNOWN or unexpected status -> wait and retry
                        Log.d(TAG, "Unknown task status: ${taskResponse.taskStatus}, retrying in 5s")
                        runBlocking { delay(5000L) }
                        continue
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching task", e)
                if (e is NVFlareError.ServerRequestedStop) {
                    return Pair(null, true)
                }
                // Retry after delay
                runBlocking { delay(5000L) }
                continue
            }
        }
    }

    /**
     * Report a result to the server.
     */
    private fun reportResult(ctx: Context, output: DXO): Boolean {
        Log.d(TAG, "reportResult: Starting to report result")
        
        val jobId = currentJobId ?: run {
            Log.e(TAG, "reportResult: No current job ID")
            return false
        }
        val taskId = ctx.get(ContextKey.TASK_ID) as? String ?: run {
            Log.e(TAG, "reportResult: No task ID in context")
            return false
        }
        val taskName = ctx.get(ContextKey.TASK_NAME) as? String ?: run {
            Log.e(TAG, "reportResult: No task name in context")
            return false
        }
        
        Log.d(TAG, "reportResult: Reporting result for job=$jobId, task=$taskId, name=$taskName")
        Log.d(TAG, "reportResult: Output DXO has ${output.toMap().size} keys: ${output.toMap().keys}")
        
        try {
            Log.d(TAG, "reportResult: Calling connection.sendResult...")
            val resultResponse = runBlocking {
                connection.sendResult(
                    jobId = jobId,
                    taskId = taskId,
                    taskName = taskName,
                    weightDiff = output.toMap()
                )
            }
            
            Log.d(TAG, "reportResult: Got response from sendResult: $resultResponse")
            
            when (resultResponse.status) {
                "OK" -> {
                    Log.d(TAG, "Result reported successfully")
                    return true
                }
                "RETRY" -> {
                    Log.d(TAG, "Result report retry requested, waiting 5000ms")
                    runBlocking { delay(5000L) }
                    return reportResult(ctx, output) // Retry
                }
                else -> {
                    Log.e(TAG, "Result report failed with status: ${resultResponse.status}")
                    return false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error reporting result", e)
            return false
        }
    }

    /**
     * Apply filters to data.
     */
    private fun applyFilters(data: Any, filters: List<Filter>, ctx: Context): Any {
        var result = data
        for (filter in filters) {
            if (result is DXO) {
                result = filter.filter(result, ctx, abortSignal)
            }
        }
        return result
    }
    
    /**
     * Emit progress update to the app.
     * Sends TrainingProgress objects with detailed status information.
     */
    private fun emitProgress(progress: TrainingProgress) {
        Log.d(TAG, "Progress: ${progress.phase} - ${progress.message}")
        onProgressUpdate?.invoke(progress)
    }
} 