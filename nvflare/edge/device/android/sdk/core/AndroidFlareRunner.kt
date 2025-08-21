package com.nvidia.nvflare.sdk.core

import android.content.Context as AndroidContext
import android.util.Log
import com.nvidia.nvflare.sdk.core.Connection
import com.nvidia.nvflare.sdk.core.JobResponse
import com.nvidia.nvflare.sdk.core.TaskResponse
import com.nvidia.nvflare.sdk.core.ResultResponse
import com.nvidia.nvflare.sdk.core.NVFlareError
import com.nvidia.nvflare.sdk.utils.asMap
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.ContextKey
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Filter
import com.nvidia.nvflare.sdk.core.NoOpFilter
import com.nvidia.nvflare.sdk.core.NoOpEventHandler
import com.nvidia.nvflare.sdk.core.NoOpTransform
import com.nvidia.nvflare.sdk.core.SimpleBatch
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.AndroidExecutor
import com.nvidia.nvflare.sdk.TrainerRegistry
import com.nvidia.nvflare.sdk.training.ETTrainer

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay

/**
 * Android-specific implementation of FlareRunner.
 * Bridges the old Connection-based approach with the new SDK architecture.
 * Implements all required abstract methods from FlareRunner base class.
 */
class AndroidFlareRunner(
    private val context: AndroidContext,
    private val connection: Connection,
    jobName: String,  // Add job_name parameter
    dataSource: DataSource,
    deviceInfo: Map<String, String>,
    userInfo: Map<String, String>,
    jobTimeout: Float,
    inFilters: List<Filter>? = null,
    outFilters: List<Filter>? = null,
    resolverRegistry: Map<String, Class<*>>? = null
) : FlareRunner(
    jobName = jobName,  // Pass job_name to parent
    dataSource = dataSource,
    deviceInfo = deviceInfo,
    userInfo = userInfo,
    jobTimeout = jobTimeout,
    inFilters = inFilters,
    outFilters = outFilters,
    resolverRegistry = resolverRegistry
) {
    private val TAG = "AndroidFlareRunner"
    private var currentJobId: String? = null

    override fun addBuiltinResolvers() {
        // Add Android-specific component resolvers here
        // Follow the Python pattern: component_type -> class_reference
        resolverRegistryMap.putAll(mapOf(
            "Executor.AndroidExecutor" to AndroidExecutor::class.java,
            "Trainer.DLTrainer" to AndroidExecutor::class.java,  // Map Trainer.DLTrainer to AndroidExecutor
            "Filter.NoOpFilter" to NoOpFilter::class.java,
            "EventHandler.NoOpEventHandler" to NoOpEventHandler::class.java,
            "Transform.NoOpTransform" to NoOpTransform::class.java,
            "Batch.SimpleBatch" to SimpleBatch::class.java
        ))
        
        // Register trainer implementations in the dynamic registry
        TrainerRegistry.registerTrainer("cnn") { context, modelData, meta ->
            ETTrainer(context, modelData, meta.toMap())
        }
        TrainerRegistry.registerTrainer("xor") { context, modelData, meta ->
            ETTrainer(context, modelData, meta.toMap())
        }
        // Future trainers can be added here without code changes to AndroidExecutorFactory
        
        // Note: datasource resolver is provided by the app
    }

    override fun getAndroidContext(): android.content.Context {
        return context
    }

    override fun getJob(ctx: Context, abortSignal: Signal): Map<String, Any>? {
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
                            "job_name" to jobName,
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

    override fun getTask(ctx: Context, abortSignal: Signal): Pair<Map<String, Any>?, Boolean> {
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

                when (taskResponse.taskStatus) {
                    TaskResponse.TaskStatus.OK -> {
                        // Convert TaskResponse to the format expected by FlareRunner
                        // Extract model data properly to avoid corruption
                        val modelData = when (taskResponse.taskData?.data) {
                            is com.google.gson.JsonPrimitive -> {
                                // Try to get as string first, fallback to toString if it fails
                                try {
                                    taskResponse.taskData.data.asString
                                } catch (e: Exception) {
                                    // If asString fails, use toString (this might corrupt binary data)
                                    taskResponse.taskData.data.toString()
                                }
                            }
                            is com.google.gson.JsonObject -> {
                                // For JSON objects, convert to string
                                taskResponse.taskData.data.toString()
                            }
                            else -> {
                                // For other types, try to get as string or fallback
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
                    TaskResponse.TaskStatus.DONE -> {
                        Log.d(TAG, "Task session completed")
                        return Pair(null, true)
                    }
                    TaskResponse.TaskStatus.ERROR -> {
                        Log.d(TAG, "Task fetch error: ${taskResponse.message}")
                        runBlocking { delay(5000L) }
                        continue
                    }
                    TaskResponse.TaskStatus.RETRY -> {
                        val retryWait = taskResponse.retryWait ?: 5000L
                        Log.d(TAG, "Task fetch retry requested, waiting ${retryWait}ms")
                        runBlocking { delay(retryWait.toLong()) }
                        continue
                    }
                    else -> {
                        Log.d(TAG, "Task fetch failed with status: ${taskResponse.taskStatus}")
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

    override fun reportResult(ctx: Context, output: DXO): Boolean {
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
} 