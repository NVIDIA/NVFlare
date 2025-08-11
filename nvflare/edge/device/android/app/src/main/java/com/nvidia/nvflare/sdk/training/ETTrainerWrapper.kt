package com.nvidia.nvflare.sdk.training

import android.util.Log
import com.nvidia.nvflare.sdk.core.NVFlareError
import com.nvidia.nvflare.sdk.models.TrainingConfig
import com.nvidia.nvflare.sdk.training.ETTrainer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class ETTrainerWrapper(
    private val context: android.content.Context,
    private val modelBase64: String,
    private val meta: TrainingConfig
) : Trainer {
    private val TAG = "ETTrainerWrapper"
    private val trainer: ETTrainer

    init {
        Log.d(TAG, "ETTrainerWrapper: Initializing with model and meta")
        trainer = ETTrainer(context, modelBase64, meta.toMap())
        Log.d(TAG, "ETTrainerWrapper: Initialization complete")
    }

    override suspend fun train(config: TrainingConfig, modelData: String?): Map<String, Any> = withContext(Dispatchers.IO) {
        Log.d(TAG, "ETTrainerWrapper: Starting train()")
        Log.d(TAG, "ETTrainerWrapper: Received modelData length: ${modelData?.length ?: 0}")
        Log.d(TAG, "ETTrainerWrapper: Received modelData starts with '{': ${modelData?.startsWith("{")}")
        Log.d(TAG, "ETTrainerWrapper: Using modelBase64 length: ${modelBase64.length}")
        Log.d(TAG, "ETTrainerWrapper: Using modelBase64 starts with '{': ${modelBase64.startsWith("{")}")
        try {
            val result = trainer.train(config, modelData)
            Log.d(TAG, "ETTrainerWrapper: train() completed with result keys: ${result.keys}")
            result
        } catch (e: Exception) {
            Log.e(TAG, "ETTrainerWrapper: Error during training", e)
            throw NVFlareError.TrainingFailed("Training failed: ${e.message}")
        }
    }
}
