package com.nvidia.nvflare.training

import android.util.Log
import com.nvidia.nvflare.models.NVFlareError
import com.nvidia.nvflare.models.TrainingConfig
import com.nvidia.nvflare.trainer.ETTrainer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ETTrainerWrapper(
    private val modelBase64: String,
    private val meta: TrainingConfig
) : Trainer {
    private val TAG = "ETTrainerWrapper"
    private val trainer: ETTrainer

    init {
        Log.d(TAG, "ETTrainerWrapper: Initializing with model and meta")
        trainer = ETTrainer(modelBase64, meta.toMap())
        Log.d(TAG, "ETTrainerWrapper: Initialization complete")
    }

    override suspend fun train(config: TrainingConfig): Map<String, Any> = withContext(Dispatchers.IO) {
        Log.d(TAG, "ETTrainerWrapper: Starting train()")
        try {
            val result = trainer.train(config)
            Log.d(TAG, "ETTrainerWrapper: train() completed with result keys: ${result.keys}")
            result
        } catch (e: Exception) {
            Log.e(TAG, "ETTrainerWrapper: Error during training", e)
            throw NVFlareError.TrainingFailed("Training failed: ${e.message}")
        }
    }
} 