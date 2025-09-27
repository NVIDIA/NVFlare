package com.nvidia.nvflare.sdk.utils

// Model Native Formats
enum class ModelNativeFormat {
    BINARY,
    JSON,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelNativeFormat {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

// Model Encodings
enum class ModelEncoding {
    BASE64,
    RAW,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelEncoding {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

// Task Headers
object TaskHeaderKey {
    const val TASK_SEQ = "task_seq"
    const val UPDATE_INTERVAL = "update_interval"
    const val CURRENT_ROUND = "current_round"
    const val NUM_ROUNDS = "num_rounds"
    const val CONTRIBUTION_ROUND = "contribution_round"
    const val JOB_NAME = "job_name"
} 