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
    dataSource: DataSource,
    deviceInfo: Map<String, String>,
    userInfo: Map<String, String>,
    jobTimeout: Float,
    inFilters: List<Filter>? = null,
    outFilters: List<Filter>? = null,
    resolverRegistry: Map<String, Class<*>>? = null
) : FlareRunner(
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
    private var currentJobName: String? = null

    override fun addBuiltinResolvers() {
        // Add Android-specific component resolvers here
        // Follow the Python pattern: component_type -> class_reference
        resolverRegistryMap.putAll(mapOf(
            "Executor.AndroidExecutor" to AndroidExecutor::class.java,
            "Filter.NoOpFilter" to NoOpFilter::class.java,
            "EventHandler.NoOpEventHandler" to NoOpEventHandler::class.java,
            "Transform.NoOpTransform" to NoOpTransform::class.java,
            "Batch.SimpleBatch" to SimpleBatch::class.java
        ))
        // Note: datasource resolver is provided by the app
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
                    connection.fetchJob()
                }

                when (jobResponse.status) {
                    "stopped" -> {
                        Log.d(TAG, "Server requested stop")
                        return null
                    }
                    "OK" -> {
                        currentJobId = jobResponse.jobId
                        currentJobName = jobResponse.jobName
                        
                        // Convert JobResponse to the format expected by FlareRunner
                        return mapOf(
                            "job_id" to (jobResponse.jobId ?: ""),
                            "job_name" to (jobResponse.jobName ?: ""),
                            "job_data" to (jobResponse.jobData?.asMap() ?: emptyMap<String, Any>())
                        )
                    }
                    "RETRY" -> {
                        val retryWait = jobResponse.retryWait ?: 5000L
                        Log.d(TAG, "Server requested retry, waiting ${retryWait}ms")
                        runBlocking { delay(retryWait) }
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
                runBlocking { delay(5000) }
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
                                "data" to (taskResponse.taskData?.data?.toString() ?: ""),
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
                    TaskResponse.TaskStatus.RETRY -> {
                        val retryWait = taskResponse.retryWait ?: 5000L
                        Log.d(TAG, "Server requested task retry, waiting ${retryWait}ms")
                        runBlocking { delay(retryWait) }
                        continue
                    }
                    TaskResponse.TaskStatus.NO_TASK -> {
                        val retryWait = taskResponse.retryWait ?: 5000L
                        Log.d(TAG, "No tasks available, retrying in ${retryWait}ms")
                        runBlocking { delay(retryWait) }
                        continue
                    }
                    else -> {
                        if (!taskResponse.taskStatus.shouldContinueTraining) {
                            Log.d(TAG, "Task fetch failed, retrying...")
                            runBlocking { delay(5000) }
                        }
                        continue
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching task", e)
                runBlocking { delay(5000) }
                continue
            }
        }
    }

    override fun reportResult(result: Map<String, Any>, ctx: Context, abortSignal: Signal): Boolean {
        if (abortSignal.isTriggered) {
            return true
        }

        val jobId = currentJobId ?: return true
        val taskId = ctx.get(ContextKey.TASK_ID) as? String ?: return true
        val taskName = ctx.get(ContextKey.TASK_NAME) as? String ?: return true

        return try {
            val resultResponse = runBlocking {
                connection.sendResult(
                    jobId = jobId,
                    taskId = taskId,
                    taskName = taskName,
                    weightDiff = result
                )
            }

            when (resultResponse.status) {
                "OK" -> {
                    Log.d(TAG, "Result sent successfully")
                    false // Continue with more tasks
                }
                else -> {
                    Log.e(TAG, "Failed to send result: ${resultResponse.message}")
                    true // Session done due to error
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error sending result", e)
            true // Session done due to error
        }
    }
} 