package com.nvidia.nvflare.sdk.core

import com.google.gson.annotations.SerializedName
import com.nvidia.nvflare.sdk.models.Job
import com.google.gson.JsonObject

data class JobResponse(
    @SerializedName("status")
    val status: String,
    
    @SerializedName("job_id")
    val jobId: String?,
    
    @SerializedName("job_name")
    val jobName: String?,
    
    @SerializedName("job_data")
    val jobData: JsonObject?,
    
    @SerializedName("method")
    val method: String?,
    
    @SerializedName("retry_wait")
    val retryWait: Int?,
    
    @SerializedName("message")
    val message: String?,
    
    @SerializedName("details")
    val details: Map<String, String>?
) {
    fun toJob(): Job {
        if (jobId == null) {
            throw NVFlareError.InvalidRequest("Can't convert JobResponse to Job")
        }
        return Job(id = jobId, status = "running")
    }
} 