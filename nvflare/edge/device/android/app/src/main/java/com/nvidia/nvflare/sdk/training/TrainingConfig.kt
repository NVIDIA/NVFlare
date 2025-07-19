package com.nvidia.nvflare.sdk.training

import com.nvidia.nvflare.models.DatasetType
import com.nvidia.nvflare.models.MetaKey

data class TrainingConfig(
    val totalEpochs: Int = 1,
    val batchSize: Int = 4,
    val learningRate: Float = 0.1f,
    val method: String = "cnn",
    val dataSetType: String = DatasetType.CIFAR10,
    val kind: String? = null
) {
    companion object {
        fun fromMap(data: Map<String, Any>): TrainingConfig {
            return TrainingConfig(
                totalEpochs = (data[MetaKey.TOTAL_EPOCHS] as? Number)?.toInt() ?: 1,
                batchSize = (data[MetaKey.BATCH_SIZE] as? Number)?.toInt() ?: 4,
                learningRate = (data[MetaKey.LEARNING_RATE] as? Number)?.toFloat() ?: 0.1f,
                method = data["method"] as? String ?: "xor",
                dataSetType = data[MetaKey.DATASET_TYPE] as? String ?: DatasetType.XOR,
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
