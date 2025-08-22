package com.nvidia.nvflare.sdk.training

import com.nvidia.nvflare.sdk.models.TrainingConfig

import com.nvidia.nvflare.sdk.core.Dataset

interface Trainer {
    suspend fun train(config: TrainingConfig, dataset: Dataset, modelData: String? = null): Map<String, Any>
}
