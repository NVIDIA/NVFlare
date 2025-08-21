package com.nvidia.nvflare.sdk.training

import com.nvidia.nvflare.sdk.models.TrainingConfig

interface Trainer {
    suspend fun train(config: TrainingConfig, modelData: String? = null): Map<String, Any>
}
