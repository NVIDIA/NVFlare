package com.nvidia.nvflare.sdk.core

sealed class NVFlareError : Exception() {
    // Network related
    data class JobFetchFailed(override val message: String = "Failed to fetch job") : NVFlareError()
    data class TaskFetchFailed(override val message: String) : NVFlareError()
    data class InvalidRequest(override val message: String) : NVFlareError()
    data class AuthError(override val message: String) : NVFlareError()
    data class ServerError(override val message: String) : NVFlareError()
    data class NetworkError(override val message: String) : NVFlareError()

    // Training related
    data class InvalidMetadata(override val message: String) : NVFlareError()
    data class InvalidModelData(override val message: String) : NVFlareError()
    data class TrainingFailed(override val message: String) : NVFlareError()
    object ServerRequestedStop : NVFlareError() {
        override val message: String = "Server requested stop"
    }
}
