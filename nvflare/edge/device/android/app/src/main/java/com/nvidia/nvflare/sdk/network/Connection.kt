package com.nvidia.nvflare.sdk.network

import android.content.Context
import android.util.Log
import androidx.lifecycle.MutableLiveData
import com.nvidia.nvflare.sdk.network.JobResponse
import com.nvidia.nvflare.sdk.network.TaskResponse
import com.nvidia.nvflare.sdk.network.ResultResponse
import com.nvidia.nvflare.sdk.network.NVFlareError
import com.nvidia.nvflare.sdk.utils.JSONValue
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
    private val TAG = "Connection"
    private var currentCookie: JSONValue? = null
    private var capabilities: Map<String, Any> = mapOf("methods" to emptyList<String>())
    private val gson = Gson()
    private val httpClient = OkHttpClient()

    // Add hostname and port properties to match iOS
    val hostname = MutableLiveData<String>("")
    val port = MutableLiveData<Int>(0)

    val isValid: Boolean
        get() = hostname.value?.isNotEmpty() == true && (port.value ?: 0) > 0 && (port.value
            ?: 0) <= 65535

    // Device info matching iOS exactly
    private val deviceId: String =
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
            context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode.toString()
        } else {
            context.packageManager.getPackageInfo(context.packageName, 0).versionCode.toString()
        }
    private val deviceInfo: Map<String, String> = mapOf(
        "device_id" to deviceId,
        "app_name" to "test",
        "app_version" to context.packageManager.getPackageInfo(context.packageName, 0).versionName,
        "platform" to "android",
        "platform_version" to "1.2.2"
    )
    
    // User info to match protocol test configuration
    private val userInfo: Map<String, String> = mapOf(
        "user_id" to "xyz"
    )

    fun setCapabilities(capabilities: Map<String, Any>) {
        this.capabilities = capabilities
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
            .scheme("http")
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment("job")
            .build()

        // Prepare request body with job_name instead of capabilities
        val requestBody = JsonObject().apply {
            add("job_name", JsonPrimitive(jobName))
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
            .header("X-Flare-Device-ID", deviceId)
            .header("X-Flare-Device-Info", infoToQueryString(deviceInfo))
            .header("X-Flare-User-Info", infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string()
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            Log.d(TAG, "Response body: $responseBody")

            // Check status code first like iOS
            when (response.code) {
                200 -> {
                    val jobResponse = gson.fromJson(responseBody, JobResponse::class.java)
                    when (jobResponse.status) {
                        "OK" -> jobResponse
                        "RETRY" -> {
                            val retryWait = jobResponse.retryWait ?: 5000
                            Log.d(TAG, "Retrying job fetch after $retryWait ms")
                            delay(retryWait.toLong())
                            fetchJob(jobName)
                        }

                        else -> throw NVFlareError.JobFetchFailed("Job fetch failed")
                    }
                }

                400 -> throw NVFlareError.InvalidRequest("Invalid request")
                403 -> throw NVFlareError.AuthError("Authentication error")
                500 -> throw NVFlareError.ServerError("Server error")
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
            .scheme("http")
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment("task")
            .addQueryParameter("job_id", jobId)
            .build()

        // Prepare request body with cookie
        val requestBody = if (currentCookie != null) {
            JsonObject().apply {
                add("cookie", gson.toJsonTree(currentCookie?.toAny()))
            }
        } else {
            JsonObject() // Empty JSON object like iOS
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
            .header("X-Flare-Device-ID", deviceId)
            .header("X-Flare-Device-Info", infoToQueryString(deviceInfo))
            .header("X-Flare-User-Info", infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string()
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            Log.d(TAG, "Response body: $responseBody")

            // Check status code first like iOS
            when (response.code) {
                200 -> {
                    val taskResponse = gson.fromJson(responseBody, TaskResponse::class.java)
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

                400 -> throw NVFlareError.InvalidRequest("Invalid request")
                403 -> throw NVFlareError.AuthError("Authentication error")
                500 -> throw NVFlareError.ServerError("Server error")
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
        if (!isValid) {
            throw NVFlareError.InvalidRequest("Invalid hostname or port")
        }

        val url = HttpUrl.Builder()
            .scheme("http")
            .host(hostname.value ?: "")
            .port(port.value ?: 0)
            .addPathSegment("result")
            .build()

        // Prepare request body
        val requestBody = JsonObject().apply {
            addProperty("job_id", jobId)
            addProperty("task_id", taskId)
            addProperty("task_name", taskName)
            add("result", gson.toJsonTree(weightDiff))
            if (currentCookie != null) {
                add("cookie", gson.toJsonTree(currentCookie?.toAny()))
            }
        }

        val request = Request.Builder()
            .url(url)
            .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
            .header("X-Flare-Device-ID", deviceId)
            .header("X-Flare-Device-Info", infoToQueryString(deviceInfo))
            .header("X-Flare-User-Info", infoToQueryString(userInfo))
            .build()

        Log.d(TAG, "Sending request: ${request.method} ${request.url}")
        Log.d(TAG, "Headers: ${request.headers}")
        Log.d(TAG, "Request body: $requestBody")

        try {
            val response = httpClient.newCall(request).execute()
            val responseBody = response.body?.string()
            Log.d(TAG, "Response Status Code: ${response.code}")
            Log.d(TAG, "Response Headers: ${response.headers}")
            Log.d(TAG, "Response body: $responseBody")

            // Check status code first like iOS
            when (response.code) {
                200 -> {
                    val resultResponse = gson.fromJson(responseBody, ResultResponse::class.java)
                    when (resultResponse.status) {
                        "OK" -> resultResponse
                        else -> throw NVFlareError.TrainingFailed(
                            resultResponse.message ?: "Unknown error"
                        )
                    }
                }

                400 -> throw NVFlareError.InvalidRequest("Invalid request")
                403 -> throw NVFlareError.AuthError("Authentication error")
                500 -> throw NVFlareError.ServerError("Server error")
                else -> throw NVFlareError.TrainingFailed("Result send failed")
            }
        } catch (e: IOException) {
            Log.e(TAG, "Network error sending result", e)
            throw NVFlareError.NetworkError("Network error")
        }
    }
}