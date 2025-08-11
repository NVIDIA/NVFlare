package com.nvidia.nvflare.sdk.core

import android.util.Log
import com.nvidia.nvflare.sdk.config.processTrainConfig
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.ContextKey
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Filter
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.core.EventType
import com.nvidia.nvflare.sdk.core.Executor

/**
 * Main orchestrator for federated learning on edge devices.
 * Handles job fetching, task execution, and result reporting.
 */
abstract class FlareRunner(
    protected val jobName: String,  // Make protected and remove duplicate
    private val dataSource: DataSource,
    private val deviceInfo: Map<String, String>,
    private val userInfo: Map<String, String>,
    protected val jobTimeout: Float,  // Make protected so subclasses can access
    private val inFilters: List<Filter>? = null,
    private val outFilters: List<Filter>? = null,
    private val resolverRegistry: Map<String, Class<*>>? = null
) {
    private val TAG = "FlareRunner"
    protected val abortSignal = Signal()
    protected var jobId: String? = null
    protected var cookie: Any? = null

    protected val resolverRegistryMap = mutableMapOf<String, Class<*>>()

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
     * Override this in platform-specific implementations.
     */
    protected open fun addBuiltinResolvers() {
        // Default implementation is empty
    }

    /**
     * Get Android context for platform-specific operations.
     * Override this in platform-specific implementations.
     */
    protected open fun getAndroidContext(): android.content.Context {
        throw UnsupportedOperationException("getAndroidContext() not implemented in base FlareRunner")
    }

    /**
     * Main run loop that continuously processes jobs.
     */
    fun run() {
        Log.d(TAG, "Starting FlareRunner")
        while (!abortSignal.isTriggered) {
            val sessionDone = doOneJob()
            if (sessionDone) {
                Log.d(TAG, "Session completed")
                break
            }
        }
        Log.d(TAG, "FlareRunner stopped")
    }

    /**
     * Stop the runner.
     */
    fun stop() {
        Log.d(TAG, "Stopping FlareRunner")
        abortSignal.trigger(null)
    }

    /**
     * Process one complete job.
     */
    private fun doOneJob(): Boolean {
        val ctx = Context()
        ctx[ContextKey.RUNNER] = this
        ctx[ContextKey.DATA_SOURCE] = dataSource

        // Try to get a job
        val job = getJob(ctx, abortSignal)
        if (job == null) {
            Log.d(TAG, "No job available")
            return true
        }

        jobId = job["job_id"] as? String
        val jobData = job["job_data"] as? Map<String, Any>

        Log.d(TAG, "Processing job: $jobName (ID: $jobId)")

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
            processTrainConfig(getAndroidContext(), config, resolverRegistryMap)
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

        // Process tasks
        while (!abortSignal.isTriggered) {
            val (task, sessionDone) = getTask(ctx, abortSignal)
            
            if (abortSignal.isTriggered) {
                return true
            }

            if (task == null) {
                Log.d(TAG, "No more tasks for this job")
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

            // Apply input filters
            var filteredTaskDxo = applyFilters(taskDxo, inputFilters, taskCtx)
            if (filteredTaskDxo !is DXO) {
                throw RuntimeException("Task data after filtering is not valid DXO: ${filteredTaskDxo::class.java}")
            }

            if (abortSignal.isTriggered) {
                return true
            }

            // Execute task
            taskCtx.fireEvent(EventType.BEFORE_TRAIN, System.currentTimeMillis(), abortSignal)
            val output = executor.execute(filteredTaskDxo, taskCtx, abortSignal)

            if (output !is DXO) {
                throw RuntimeException("Output from ${executor::class.java} is not a valid DXO: ${output::class.java}")
            }

            taskCtx.fireEvent(EventType.AFTER_TRAIN, Pair(System.currentTimeMillis(), output), abortSignal)

            if (abortSignal.isTriggered) {
                return true
            }

            // Apply output filters
            var filteredOutput = applyFilters(output, outputFilters, taskCtx)
            if (filteredOutput !is DXO) {
                throw RuntimeException("Output after filtering is not valid DXO: ${filteredOutput::class.java}")
            }

            // Report result
            val result = reportResult(ctx, filteredOutput)
            if (!result) {
                Log.e(TAG, "Failed to report result")
                return true
            }
        }

        return false
    }

    /**
     * Get a job from the server.
     */
    protected abstract fun getJob(ctx: Context, abortSignal: Signal): Map<String, Any>?

    /**
     * Get a task from the server.
     */
    protected abstract fun getTask(ctx: Context, abortSignal: Signal): Pair<Map<String, Any>?, Boolean>

    /**
     * Report a result to the server.
     */
    protected abstract fun reportResult(ctx: Context, output: DXO): Boolean

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
} 