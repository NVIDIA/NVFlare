package com.nvidia.nvflare.sdk.training

import com.google.gson.annotations.SerializedName
import com.nvidia.nvflare.sdk.utils.TaskHeaderKey

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

// Training Types
enum class TrainerType {
    EXECUTORCH
}

enum class MethodType(val displayName: String) {
    CNN("cnn"),
    XOR("xor");

    val requiredDataset: String
        get() = when (this) {
            CNN -> "cifar10"
            XOR -> "xor"
        }

    companion object {
        fun fromString(value: String): MethodType? {
            return values().find { it.displayName.equals(value, ignoreCase = true) }
        }
    }
}

enum class TrainingStatus {
    IDLE,
    TRAINING,
    STOPPING
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

// Job
data class Job(
    val id: String,
    val status: String
)

// Training Task
data class TrainingTask(
    val id: String,
    val name: String,
    val jobId: String,
    val modelData: String,
    val trainingConfig: TrainingConfig,
    val currentRound: Int = 0,
    val numRounds: Int = 1,
    val updateInterval: Float = 1.0f
)

// Training Configuration
data class TrainingConfig(
    val totalEpochs: Int = 1,
    val batchSize: Int = 4,
    val learningRate: Float = 0.1f,
    val method: String = "cnn",
    val dataSetType: String = DatasetType.CIFAR10,
    val kind: String? = null
) {
    init {
        // Validate that batch size is appropriate for the method/dataset
        if (method == "xor" && batchSize > 4) {
            throw IllegalArgumentException("XOR method requires batch size <= 4 (dataset has only 4 samples)")
        }
    }
    companion object {
        private fun determineMethodFromJobName(data: Map<String, Any>): String {
            // Try to determine method from job name or other context
            val jobName = data[TaskHeaderKey.JOB_NAME] as? String ?: ""
            return when {
                jobName.lowercase().contains("cifar") || jobName.lowercase().contains("cnn") -> "cnn"
                jobName.lowercase().contains("xor") -> "xor"
                else -> "xor" // Default fallback
            }
        }
        
        fun fromMap(data: Map<String, Any>): TrainingConfig {
            val method = data["method"] as? String ?: determineMethodFromJobName(data)
            val dataSetType = data[MetaKey.DATASET_TYPE] as? String ?: DatasetType.XOR
            
            // Use batch size 1 for XOR (small dataset), 4 for CNN (larger dataset)
            val defaultBatchSize = when {
                method == "xor" || dataSetType == DatasetType.XOR -> 1
                else -> 4
            }
            
            return TrainingConfig(
                totalEpochs = (data[MetaKey.TOTAL_EPOCHS] as? Number)?.toInt() ?: 1,
                batchSize = (data[MetaKey.BATCH_SIZE] as? Number)?.toInt() ?: defaultBatchSize,
                learningRate = (data[MetaKey.LEARNING_RATE] as? Number)?.toFloat() ?: 0.1f,
                method = method,
                dataSetType = dataSetType,
                kind = data["kind"] as? String
            )
        }
    }

    fun toMap(): Map<String, Any> = mapOf(
        MetaKey.TOTAL_EPOCHS to totalEpochs,
        MetaKey.BATCH_SIZE to batchSize,
        MetaKey.LEARNING_RATE to learningRate,
        "method" to method,
        MetaKey.DATASET_TYPE to dataSetType
    ).let { map ->
        if (kind != null) {
            map + ("kind" to kind)
        } else {
            map
        }
    }
}
