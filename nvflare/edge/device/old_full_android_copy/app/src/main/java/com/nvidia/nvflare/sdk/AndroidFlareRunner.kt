package com.nvidia.nvflare.sdk

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.connection.Connection
import com.nvidia.nvflare.models.JobResponse
import com.nvidia.nvflare.models.TaskResponse
import com.nvidia.nvflare.models.ResultResponse
import com.nvidia.nvflare.models.NVFlareError
import com.nvidia.nvflare.models.asMap
import com.nvidia.nvflare.sdk.defs.*
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.delay

/**
 * Android-specific implementation of FlareRunner.
 * Bridges the old Connection-based approach with the new SDK architecture.
 */
class AndroidFlareRunner(
    private val context: Context,
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
        // For now, we'll rely on the app-provided resolvers
    }

    override fun getJob(ctx: Context, abortSignal: Signal): Map<String, Any>? {
        if (abortSignal.isTriggered) {
            return null
        }

        return try {
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
                    mapOf(
                        "job_id" to (jobResponse.jobId ?: ""),
                        "job_name" to (jobResponse.jobName ?: ""),
                        "job_data" to (jobResponse.jobData?.asMap() ?: emptyMap<String, Any>())
                    )
                }
                else -> {
                    Log.d(TAG, "Job fetch failed with status: ${jobResponse.status}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching job", e)
            if (e is NVFlareError.ServerRequestedStop) {
                return null
            }
            // Add delay before retry
            runBlocking { delay(5000) }
            null
        }
    }

    override fun getTask(ctx: Context, abortSignal: Signal): Pair<Map<String, Any>?, Boolean> {
        if (abortSignal.isTriggered) {
            return Pair(null, true)
        }

        val jobId = currentJobId ?: return Pair(null, true)

        return try {
            val taskResponse = runBlocking {
                connection.fetchTask(jobId)
            }

            when (taskResponse.taskStatus) {
                TaskResponse.TaskStatus.OK -> {
                    // Convert TaskResponse to the format expected by FlareRunner
                    val taskMap = mapOf(
                        "task_id" to (taskResponse.taskId ?: ""),
                        "task_name" to (taskResponse.taskName ?: ""),
                        "task_data" to mapOf(
                            "data" to (taskResponse.taskData?.data?.toString() ?: ""),
                            "meta" to (taskResponse.taskData?.meta?.asMap() ?: emptyMap<String, Any>()),
                            "kind" to (taskResponse.taskData?.kind ?: "")
                        ),
                        "cookie" to taskResponse.cookie?.asMap()
                    )
                    Pair(taskMap, false)
                }
                TaskResponse.TaskStatus.DONE -> {
                    Log.d(TAG, "Task session completed")
                    Pair(null, true)
                }
                else -> {
                    if (!taskResponse.taskStatus.shouldContinueTraining) {
                        Log.d(TAG, "No tasks available, retrying...")
                        runBlocking { delay(5000) }
                    }
                    Pair(null, false)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching task", e)
            runBlocking { delay(5000) }
            Pair(null, false)
        }
    }

    override fun reportResult(result: Map<String, Any>, ctx: Context, abortSignal: Signal): Boolean {
        if (abortSignal.isTriggered) {
            return true
        }

        val jobId = currentJobId ?: return true
        val taskId = ctx[ContextKey.TASK_ID] as? String ?: return true
        val taskName = ctx[ContextKey.TASK_NAME] as? String ?: return true

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