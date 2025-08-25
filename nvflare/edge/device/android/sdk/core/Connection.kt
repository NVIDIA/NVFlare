package com.nvidia.nvflare.sdk.core

import android.content.Context
import android.util.Log
import androidx.lifecycle.MutableLiveData
import com.nvidia.nvflare.sdk.models.JobResponse
import com.nvidia.nvflare.sdk.models.TaskResponse
import com.nvidia.nvflare.sdk.models.ResultResponse
import com.nvidia.nvflare.sdk.core.NVFlareError
import com.nvidia.nvflare.sdk.models.JSONValue
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.IOException
import com.google.gson.Gson
import com.google.gson.JsonObject
import okhttp3.HttpUrl
import com.google.gson.JsonPrimitive

class Connection(private val context: Context) {
    companion object {
        // API endpoints
        private const val ENDPOINT_JOB = "job"
        private const val ENDPOINT_TASK = "task"
        private const val ENDPOINT_RESULT = "result"
        
        // Query parameter names
        private const val PARAM_JOB_ID = "job_id"
        private const val PARAM_TASK_ID = "task_id"
        private const val PARAM_TASK_NAME = "task_name"
        
        // HTTP headers
        private const val HEADER_DEVICE_ID = "X-Flare-Device-ID"
        private const val HEADER_DEVICE_INFO = "X-Flare-Device-Info"
        private const val HEADER_USER_INFO = "X-Flare-User-Info"
        
        // HTTP status codes
        private const val HTTP_OK = 200
        private const val HTTP_BAD_REQUEST = 400
        private const val HTTP_FORBIDDEN = 403
        private const val HTTP_SERVER_ERROR = 500
        
        // Response status values
        private const val STATUS_OK = "OK"
        private const val STATUS_RETRY = "RETRY"
        private const val STATUS_STOPPED = "stopped"
        
        // Content types
        private const val CONTENT_TYPE_JSON = "application/json"
        
        // HTTP scheme
        private const val HTTP_SCHEME = "http"
        
        // JSON field names
        private const val FIELD_JOB_NAME = "job_name"
        private const val FIELD_CAPABILITIES = "capabilities"
        private const val FIELD_COOKIE = "cookie"
        private const val FIELD_RESULT = "result"
        private const val FIELD_METHODS = "methods"
        
        // HTTP response headers
        private const val HEADER_CONTENT_LENGTH = "Content-Length"
    }
    
    private val TAG = "Connection"
    private var currentCookie: JSONValue? = null
    private var capabilities: Map<String, Any> = mapOf(FIELD_METHODS to emptyList<String>())
    private val gson = Gson()
    private val httpClient = OkHttpClient()

    // Add hostname and port properties to match iOS
    val hostname = MutableLiveData<String>("")
    val port = MutableLiveData<Int>(0)

    val isValid: Boolean
        get() = hostname.value?.isNotEmpty() == true && (port.value ?: 0) > 0 && (port.value
            ?: 0) <= 65535

    // Generate a stable installation ID that persists for this app installation
    private val installationId = java.util.UUID.randomUUID().toString()
    
    // Device info matching iOS exactly
    private val deviceId: String = "NVFlare_Android-$installationId"
    private var deviceInfo: Map<String, String> = mapOf(
        "device_id" to deviceId,
        "app_name" to context.packageManager.getPackageInfo(context.packageName, 0).applicationInfo.loadLabel(context.packageManager).toString(),
        "app_version" to context.packageManager.getPackageInfo(context.packageName, 0).versionName,
        "platform" to "android"
    )
    
    // User info - configurable with sensible default
    private var userInfo: Map<String, String> = mapOf(
        "user_id" to "default_user"
    )

    fun setCapabilities(capabilities: Map<String, Any>) {
        this.capabilities = capabilities
    }

    fun setUserInfo(userInfo: Map<String, String>) {
        this.userInfo = userInfo
    }

    fun getUserInfo(): Map<String, String> = userInfo

    fun getDeviceInfo(): Map<String, String> = deviceInfo

    fun setDeviceInfo(deviceInfo: Map<String, String>) {
        this.deviceInfo = deviceInfo
    }

    fun addDeviceInfo(key: String, value: String) {
        this.deviceInfo = this.deviceInfo + (key to value)
    }

    fun resetCookie() {
        currentCookie = null
    }

    private fun infoToQueryString(info: Map<String, String>): String {
        return info.map { (key, value) -> "$key=$value" }.joinToString("&")
    }

    suspend fun fetchJob(jobName: String): JobResponse = withContext(Dispatchers.IO) {
        if (!isValid) {
            throw NVFlareError.InvalidRequest("Invalid hostname or port")
        }

        val url = HttpUrl.Builder()
            .scheme(HTTP_SCHEME)
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment(ENDPOINT_JOB)
            .build()

        // Prepare request body with job_name and capabilities
        val requestBody = JsonObject().apply {
            add(FIELD_JOB_NAME, JsonPrimitive(jobName))
            add(FIELD_CAPABILITIES, gson.toJsonTree(capabilities))
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody(CONTENT_TYPE_JSON.toMediaType()))
            .header(HEADER_DEVICE_ID, deviceId)
            .header(HEADER_DEVICE_INFO, infoToQueryString(deviceInfo))
            .header(HEADER_USER_INFO, infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            
            // Log response body info without corrupting data
            val contentLength = response.headers[HEADER_CONTENT_LENGTH]?.toIntOrNull() ?: 0
            Log.d(TAG, "Response Content-Length: $contentLength bytes")
            
            // Check status code first like iOS
            when (response.code) {
                HTTP_OK -> {
                    // Parse directly from response body using charStream to avoid string conversion corruption
                    val jobResponse = if (responseBody != null) {
                        val reader = responseBody.charStream()
                        gson.fromJson(reader, JobResponse::class.java)
                    } else {
                        throw NVFlareError.JobFetchFailed("No response body")
                    }
                    
                    when (jobResponse.status) {
                        STATUS_OK -> jobResponse
                        STATUS_RETRY -> {
                            val retryWait = jobResponse.retryWait ?: 5000
                            Log.d(TAG, "Retrying job fetch after $retryWait ms")
                            delay(retryWait.toLong())
                            fetchJob(jobName)
                        }

                        else -> throw NVFlareError.JobFetchFailed("Job fetch failed")
                    }
                }

                HTTP_BAD_REQUEST -> throw NVFlareError.InvalidRequest("Invalid request")
                HTTP_FORBIDDEN -> throw NVFlareError.AuthError("Authentication error")
                HTTP_SERVER_ERROR -> throw NVFlareError.ServerError("Server error")
                else -> throw NVFlareError.JobFetchFailed("Job fetch failed")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error fetching job", e)
            throw NVFlareError.NetworkError("Network error")
        }
    }

    suspend fun fetchTask(jobId: String): TaskResponse = withContext(Dispatchers.IO) {
        if (!isValid) {
            throw NVFlareError.InvalidRequest("Invalid hostname or port")
        }

        val url = HttpUrl.Builder()
            .scheme(HTTP_SCHEME)
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment(ENDPOINT_TASK)
            .addQueryParameter(PARAM_JOB_ID, jobId)
            .build()

        // Prepare request body with cookie
        val requestBody = if (currentCookie != null) {
            JsonObject().apply {
                add(FIELD_COOKIE, gson.toJsonTree(currentCookie?.toAny()))
            }
        } else {
            JsonObject() // Empty JSON object like iOS
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody(CONTENT_TYPE_JSON.toMediaType()))
            .header(HEADER_DEVICE_ID, deviceId)
            .header(HEADER_DEVICE_INFO, infoToQueryString(deviceInfo))
            .header(HEADER_USER_INFO, infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            
            // Log response body info without corrupting data
            val contentLength = response.headers[HEADER_CONTENT_LENGTH]?.toIntOrNull() ?: 0
            Log.d(TAG, "Response Content-Length: $contentLength bytes")
            
            // Check status code first like iOS
            when (response.code) {
                HTTP_OK -> {
                    // Parse directly from response body using charStream to avoid string conversion corruption
                    val taskResponse = if (responseBody != null) {
                        val reader = responseBody.charStream()
                        gson.fromJson(reader, TaskResponse::class.java)
                    } else {
                        throw NVFlareError.TaskFetchFailed("No response body")
                    }
                    Log.d(TAG, "Parsed TaskResponse: $taskResponse")
                    
                    // Update cookie if present - convert JsonObject to JSONValue
                    taskResponse.cookie?.let { cookie ->
                        currentCookie = JSONValue.fromJsonElement(cookie)
                        Log.d(TAG, "Updated cookie: $cookie")
                    }

                    // Check task status using enum
                    val taskStatus = taskResponse.taskStatus
                    when (taskStatus) {
                        TaskResponse.TaskStatus.OK -> taskResponse
                        TaskResponse.TaskStatus.RETRY -> {
                            val retryWait = taskResponse.retryWait ?: 5000
                            Log.d(TAG, "Retrying task fetch after $retryWait ms")
                            delay(retryWait.toLong())
                            fetchTask(jobId)
                        }
                        else -> {
                            if (!taskStatus.shouldContinueTraining) {
                                throw NVFlareError.TaskFetchFailed("Task fetch failed")
                            }
                            taskResponse
                        }
                    }
                }

                HTTP_BAD_REQUEST -> throw NVFlareError.InvalidRequest("Invalid request")
                HTTP_FORBIDDEN -> throw NVFlareError.AuthError("Authentication error")
                HTTP_SERVER_ERROR -> throw NVFlareError.ServerError("Server error")
                else -> throw NVFlareError.TaskFetchFailed("Task fetch failed")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error fetching task", e)
            throw NVFlareError.NetworkError("Network error")
        }
    }

    suspend fun sendResult(
        jobId: String,
        taskId: String,
        taskName: String,
        weightDiff: Map<String, Any>
    ): ResultResponse = withContext(Dispatchers.IO) {
        Log.d(TAG, "sendResult: Starting to send result for job=$jobId, task=$taskId, name=$taskName")
        
        if (!isValid) {
            Log.e(TAG, "sendResult: Invalid hostname or port")
            throw NVFlareError.InvalidRequest("Invalid hostname or port")
        }

        val url = HttpUrl.Builder()
            .scheme(HTTP_SCHEME)
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment(ENDPOINT_RESULT)
            .addQueryParameter(PARAM_JOB_ID, jobId)
            .addQueryParameter(PARAM_TASK_ID, taskId)
            .addQueryParameter(PARAM_TASK_NAME, taskName)
            .build()

        Log.d(TAG, "sendResult: Built URL: $url")

        // Prepare request body
        val requestBody = JsonObject().apply {
            add(FIELD_RESULT, gson.toJsonTree(weightDiff))
            if (currentCookie != null) {
                add(FIELD_COOKIE, gson.toJsonTree(currentCookie?.toAny()))
            }
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody(CONTENT_TYPE_JSON.toMediaType()))
            .header(HEADER_DEVICE_ID, deviceId)
            .header(HEADER_DEVICE_INFO, infoToQueryString(deviceInfo))
            .header(HEADER_USER_INFO, infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            Log.d(TAG, "sendResult: Executing HTTP request...")
            val response = httpClient.newCall(request).execute()
            Log.d(TAG, "sendResult: HTTP response received, status: ${response.code}")
            
            val responseBody = response.body?.string()
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            Log.d(TAG, "Response body: $responseBody")

            // Check status code first like iOS
            when (response.code) {
                HTTP_OK -> {
                    Log.d(TAG, "sendResult: Parsing 200 response...")
                    val resultResponse = gson.fromJson(responseBody, ResultResponse::class.java)
                    Log.d(TAG, "sendResult: Parsed ResultResponse: $resultResponse")
                    
                    when (resultResponse.status) {
                        STATUS_OK -> {
                            Log.d(TAG, "sendResult: Result sent successfully")
                            resultResponse
                        }
                        else -> {
                            Log.e(TAG, "sendResult: Server returned error status: ${resultResponse.status}")
                            throw NVFlareError.TrainingFailed(
                                resultResponse.message ?: "Unknown error"
                            )
                        }
                    }
                }

                HTTP_BAD_REQUEST -> {
                    Log.e(TAG, "sendResult: Bad request (400)")
                    throw NVFlareError.InvalidRequest("Invalid request")
                }
                HTTP_FORBIDDEN -> {
                    Log.e(TAG, "sendResult: Authentication error (403)")
                    throw NVFlareError.AuthError("Authentication error")
                }
                HTTP_SERVER_ERROR -> {
                    Log.e(TAG, "sendResult: Server error (500)")
                    throw NVFlareError.ServerError("Server error")
                }
                else -> {
                    Log.e(TAG, "sendResult: Unexpected status code: ${response.code}")
                    throw NVFlareError.TrainingFailed("Result send failed")
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "sendResult: Network error sending result", e)
            throw NVFlareError.NetworkError("Network error")
        } catch (e: Exception) {
            Log.e(TAG, "sendResult: Unexpected error sending result", e)
            throw e
        }
    }
}