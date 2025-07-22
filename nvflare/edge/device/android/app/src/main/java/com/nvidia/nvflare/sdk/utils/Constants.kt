package com.nvidia.nvflare.sdk.utils

// Dataset Types
object DatasetType {
    const val CIFAR10 = "cifar10"
    const val XOR = "xor"
}

// Meta Keys
object MetaKey {
    const val DATASET_TYPE = "dataset_type"
    const val BATCH_SIZE = "batch_size"
    const val LEARNING_RATE = "learning_rate"
    const val TOTAL_EPOCHS = "total_epochs"
    const val DATASET_SHUFFLE = "dataset_shuffle"
}

// Model Exchange Format Constants
object ModelExchangeFormat {
    const val MODEL_BUFFER = "model_buffer"
    const val MODEL_BUFFER_TYPE = "model_buffer_type"
    const val MODEL_BUFFER_NATIVE_FORMAT = "model_buffer_native_format"
    const val MODEL_BUFFER_ENCODING = "model_buffer_encoding"
}

// Model Buffer Types
enum class ModelBufferType {
    EXECUTORCH,
    PYTORCH,
    TENSORFLOW,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelBufferType {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

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
} 