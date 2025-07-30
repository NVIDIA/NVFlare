package com.nvidia.nvflare.sdk.trainers

import com.nvidia.nvflare.app.training.ETTrainerWrapper
import com.nvidia.nvflare.app.training.Trainer
import com.nvidia.nvflare.sdk.TrainerRegistry
import com.nvidia.nvflare.sdk.training.TrainingConfig

/**
 * Factory for creating ETTrainer instances.
 * Implements the TrainerRegistry.TrainerFactory interface.
 */
class ETTrainerFactory : TrainerRegistry.TrainerFactory {
    override fun createTrainer(context: android.content.Context, modelData: String, meta: TrainingConfig): Trainer {
        return ETTrainerWrapper(context, modelData, meta)
    }
} 