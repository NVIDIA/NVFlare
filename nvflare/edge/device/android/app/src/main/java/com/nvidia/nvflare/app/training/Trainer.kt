package com.nvidia.nvflare.app.training

import com.nvidia.nvflare.sdk.training.TrainingConfig

interface Trainer {
    suspend fun train(config: TrainingConfig): Map<String, Any>
}
