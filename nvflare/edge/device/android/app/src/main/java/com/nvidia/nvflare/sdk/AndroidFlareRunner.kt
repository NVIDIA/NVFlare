package com.nvidia.nvflare.sdk

import android.content.Context as AndroidContext
import android.util.Log
import com.nvidia.nvflare.sdk.network.Connection
import com.nvidia.nvflare.sdk.network.JobResponse
import com.nvidia.nvflare.sdk.network.TaskResponse
import com.nvidia.nvflare.sdk.network.ResultResponse
import com.nvidia.nvflare.sdk.network.NVFlareError
import com.nvidia.nvflare.sdk.utils.asMap
import com.nvidia.nvflare.sdk.defs.Context
import com.nvidia.nvflare.sdk.defs.Signal
import com.nvidia.nvflare.sdk.defs.ContextKey
import com.nvidia.nvflare.sdk.defs.DataSource
import com.nvidia.nvflare.sdk.defs.Filter
import com.nvidia.nvflare.sdk.defs.NoOpFilter
import com.nvidia.nvflare.sdk.defs.NoOpEventHandler
import com.nvidia.nvflare.sdk.defs.NoOpTransform
import com.nvidia.nvflare.sdk.defs.SimpleBatch
import com.nvidia.nvflare.sdk.defs.DXO
import com.nvidia.nvflare.sdk.trainers.ETTrainerFactory

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
        TrainerRegistry.registerTrainer("cnn", ETTrainerFactory())
        TrainerRegistry.registerTrainer("xor", ETTrainerFactory())
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
                        val taskMap: Map<String, Any> = mapOf(
                            "task_id" to (taskResponse.taskId ?: ""),
                            "task_name" to (taskResponse.taskName ?: ""),
                            "task_data" to mapOf(
                                "data" to mapOf("model" to (taskResponse.taskData?.data?.toString() ?: "")),
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
        val jobId = currentJobId ?: return false
        val taskId = ctx.get(ContextKey.TASK_ID) as? String ?: return false
        val taskName = ctx.get(ContextKey.TASK_NAME) as? String ?: return false
        
        try {
            val resultResponse = runBlocking {
                connection.sendResult(
                    jobId = jobId,
                    taskId = taskId,
                    taskName = taskName,
                    weightDiff = output.toMap()
                )
            }
            
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