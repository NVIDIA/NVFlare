package com.nvidia.nvflare.training

import com.nvidia.nvflare.models.TrainingConfig

interface Trainer {
    suspend fun train(config: TrainingConfig): Map<String, Any>
} 