package com.nvidia.nvflare.sdk.models

import com.google.gson.annotations.SerializedName
import com.google.gson.JsonObject

data class ResultResponse(
    @SerializedName("status")
    val status: String,
    
    @SerializedName("task_id")
    val taskId: String?,
    
    @SerializedName("task_name")
    val taskName: String?,
    
    @SerializedName("job_id")
    val jobId: String?,
    
    @SerializedName("message")
    val message: String?,
    
    @SerializedName("details")
    val details: Map<String, String>?
) {
    companion object {
        fun fromJSON(json: JsonObject): ResultResponse {
            return ResultResponse(
                status = json.get("status").asString,
                taskId = json.get("task_id")?.asString,
                taskName = json.get("task_name")?.asString,
                jobId = json.get("job_id")?.asString,
                message = json.get("message")?.asString,
                details = json.getAsJsonObject("details")?.let { details ->
                    val map = mutableMapOf<String, String>()
                    details.entrySet().forEach { (key, value) ->
                        map[key] = value.asString
                    }
                    map
                }
            )
        }
    }
}
