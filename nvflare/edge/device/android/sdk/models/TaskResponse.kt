package com.nvidia.nvflare.sdk.models

import com.google.gson.annotations.SerializedName
import com.nvidia.nvflare.sdk.training.TrainingTask
import com.nvidia.nvflare.sdk.utils.asMap
import com.nvidia.nvflare.sdk.utils.asList
import com.nvidia.nvflare.sdk.training.TrainingConfig
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.nvidia.nvflare.sdk.core.NVFlareError

data class TaskResponse(
    @SerializedName("status")
    val status: String,
    
    @SerializedName("message")
    val message: String?,
    
    @SerializedName("job_id")
    val jobId: String?,
    
    @SerializedName("task_id")
    val taskId: String?,
    
    @SerializedName("task_name")
    val taskName: String?,
    
    @SerializedName("retry_wait")
    val retryWait: Int?,
    
    @SerializedName("task_data")
    val taskData: TaskData?,
    
    @SerializedName("cookie")
    val cookie: JsonObject?
) {
    data class TaskData(
        @SerializedName("data")
        val data: JsonElement,
        
        @SerializedName("meta")
        val meta: JsonObject?,
        
        @SerializedName("kind")
        val kind: String
    )

    enum class TaskStatus(val value: String) {
        OK("OK"),
        DONE("DONE"),
        ERROR("ERROR"),
        RETRY("RETRY"),
        NO_TASK("NO_TASK"),
        NO_JOB("NO_JOB"),
        INVALID("INVALID"),
        UNKNOWN("UNKNOWN");

        companion object {
            fun fromString(value: String): TaskStatus {
                return values().find { it.value == value } ?: UNKNOWN
            }
        }

        val isSuccess: Boolean
            get() = this == OK || this == DONE

        val shouldContinueTraining: Boolean
            get() = this == OK
        
        val isTerminal: Boolean
            get() = this == DONE || this == INVALID || this == ERROR
        
        val shouldRetryTask: Boolean
            get() = this == RETRY || this == NO_TASK
        
        val shouldLookForNewJob: Boolean
            get() = this == NO_JOB
    }

    val taskStatus: TaskStatus
        get() = TaskStatus.fromString(status)

    fun toTrainingTask(jobId: String): TrainingTask {
        if (!taskStatus.shouldContinueTraining) {
            throw NVFlareError.TaskFetchFailed(message ?: "Task status indicates training should not continue")
        }

        if (taskId == null || taskName == null || taskData == null) {
            throw NVFlareError.TaskFetchFailed("Missing required task data")
        }

        // Convert meta JsonObject to Map<String, Any>
        val metaMap = taskData.meta?.let { meta ->
            meta.entrySet().mapNotNull { (key, value) ->
                val convertedValue = when (value) {
                    is JsonPrimitive -> when {
                        value.isString -> value.asString
                        value.isNumber -> value.asNumber
                        value.isBoolean -> value.asBoolean
                        else -> null
                    }
                    is JsonObject -> value.asMap()
                    is JsonArray -> value.asList()
                    else -> null
                }
                if (convertedValue != null) key to convertedValue else null
            }.toMap()
        } ?: emptyMap()

        // Add kind to meta data
        val metaWithKind = metaMap.toMutableMap()
        metaWithKind["kind"] = taskData.kind

        // Handle both string and object data cases
        val modelData = when (taskData.data) {
            is JsonPrimitive -> taskData.data.asString
            is JsonObject -> taskData.data.toString()
            else -> throw NVFlareError.TaskFetchFailed("Unsupported data type in task_data.data")
        }

        return TrainingTask(
            id = taskId,
            name = taskName,
            jobId = jobId,
            modelData = modelData,
            trainingConfig = TrainingConfig.fromMap(metaWithKind)
        )
    }
}
