package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.config.processTrainConfig
import com.nvidia.nvflare.sdk.defs.*

/**
 * Main orchestrator for federated learning on edge devices.
 * Handles job fetching, task execution, and result reporting.
 */
abstract class FlareRunner(
    private val dataSource: DataSource,
    private val deviceInfo: Map<String, String>,
    private val userInfo: Map<String, String>,
    private val jobTimeout: Float,
    private val inFilters: List<Filter>? = null,
    private val outFilters: List<Filter>? = null,
    private val resolverRegistry: Map<String, Class<*>>? = null
) {
    private val TAG = "FlareRunner"
    protected val abortSignal = Signal()
    protected var jobId: String? = null
    protected var jobName: String? = null
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
        abortSignal.trigger()
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

        jobName = job["job_name"] as? String
        jobId = job["job_id"] as? String
        val jobData = job["job_data"] as? Map<String, Any>

        Log.d(TAG, "Processing job: $jobName (ID: $jobId)")

        // Process training configuration
        val trainConfig = if (jobData != null) {
            processTrainConfig(jobData, resolverRegistryMap)
        } else {
            throw RuntimeException("No job data available")
        }

        ctx[ContextKey.COMPONENTS] = trainConfig.objects
        ctx[ContextKey.EVENT_HANDLERS] = trainConfig.eventHandlers

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
            cookie = task["cookie"]
            val taskName = task["task_name"] as? String
            val taskData = task["task_data"] as? Map<String, Any>

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
            taskCtx[ContextKey.TASK_ID] = task["task_id"]
            taskCtx[ContextKey.TASK_NAME] = taskName
            taskCtx[ContextKey.TASK_DATA] = taskData
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
            val filteredOutput = applyFilters(output, outputFilters, taskCtx)
            if (filteredOutput !is DXO) {
                throw RuntimeException("Output after filtering for task $taskName is not a valid DXO: ${filteredOutput::class.java}")
            }

            if (abortSignal.isTriggered) {
                return true
            }

            // Report result
            val resultSessionDone = reportResult(filteredOutput.toMap(), taskCtx, abortSignal)
            if (resultSessionDone) {
                return resultSessionDone
            }

            if (abortSignal.isTriggered) {
                return true
            }
        }

        return true
    }

    /**
     * Apply filters to data.
     */
    private fun applyFilters(data: DXO, filters: List<Filter>, ctx: Context): DXO {
        var filteredData = data
        for (filter in filters) {
            filteredData = filter.filter(filteredData, ctx, abortSignal)
            if (abortSignal.isTriggered) {
                break
            }
        }
        return filteredData
    }

    /**
     * Get a job from the server.
     * Must be implemented by platform-specific subclasses.
     */
    protected abstract fun getJob(ctx: Context, abortSignal: Signal): Map<String, Any>?

    /**
     * Get a task from the server.
     * Must be implemented by platform-specific subclasses.
     */
    protected abstract fun getTask(ctx: Context, abortSignal: Signal): Pair<Map<String, Any>?, Boolean>

    /**
     * Report results to the server.
     * Must be implemented by platform-specific subclasses.
     */
    protected abstract fun reportResult(result: Map<String, Any>, ctx: Context, abortSignal: Signal): Boolean
} 