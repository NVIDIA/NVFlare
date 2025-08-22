package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.Executor
import com.nvidia.nvflare.sdk.training.Trainer
import com.nvidia.nvflare.sdk.training.ETTrainer
import com.nvidia.nvflare.sdk.models.TrainingConfig
import kotlinx.coroutines.runBlocking

/**
 * Android-specific implementation of Executor.
 * Bridges the old Trainer interface with the new SDK architecture.
 */
class AndroidExecutor(
    private val trainer: Trainer
) : Executor {
    private val TAG = "AndroidExecutor"

    override fun execute(taskData: DXO, ctx: Context, abortSignal: Signal): DXO {
        if (abortSignal.isTriggered) {
            throw RuntimeException("Execution aborted")
        }

        try {
            Log.d(TAG, "Starting training execution")
            Log.d(TAG, "Task data keys: ${taskData.data.keys}")
            Log.d(TAG, "Task data: $taskData")
            
            // Extract training configuration from task data
            val trainingConfig = extractTrainingConfig(taskData, ctx)
            
            // Extract model data from task data
            val modelData = taskData.data["model"] as? String
            Log.d(TAG, "Extracted model data: ${modelData?.take(50) ?: "null"}...")
            Log.d(TAG, "Model data length: ${modelData?.length ?: 0}")
            Log.d(TAG, "Model data starts with '{': ${modelData?.startsWith("{")}")
            Log.d(TAG, "Model data ends with '}': ${modelData?.endsWith("}")}")
            Log.d(TAG, "Model data first 100 chars: ${modelData?.take(100)}")
            Log.d(TAG, "Model data last 50 chars: ${modelData?.takeLast(50)}")
            
            // Validate model data
            if (modelData.isNullOrEmpty()) {
                Log.e(TAG, "No model data found in task data")
                throw RuntimeException("No model data found in task data")
            }
            
            // Create appropriate dataset based on training method
            val dataset = when (trainingConfig.method) {
                "xor" -> com.nvidia.nvflare.app.data.XORDataset("train")
                "cnn" -> com.nvidia.nvflare.app.data.CIFAR10Dataset(ctx.getAndroidContext())
                else -> throw IllegalArgumentException("Unsupported training method: ${trainingConfig.method}")
            }
            
            // Execute training using the Trainer interface with dataset and model data
            val result = runBlocking {
                trainer.train(trainingConfig, dataset, modelData)
            }
            
            Log.d(TAG, "Training completed successfully")
            
            // Convert result back to DXO
            return DXO.fromMap(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "Training execution failed", e)
            throw RuntimeException("Training failed: ${e.message}", e)
        }
    }
    
    private fun extractTrainingConfig(taskData: DXO, ctx: Context): TrainingConfig {
        // Extract training configuration from the task data
        val data = taskData.data
        val meta = taskData.meta
        
        // Convert meta to TrainingConfig
        val metaMap = meta as? Map<String, Any> ?: emptyMap()
        return TrainingConfig.fromMap(metaMap)
    }
}

/**
 * Factory for creating AndroidExecutor instances.
 */
object AndroidExecutorFactory {
    
    /**
     * Create an AndroidExecutor based on the training method and model data.
     */
    fun createExecutor(context: android.content.Context, method: String, modelData: String, meta: Map<String, Any>): AndroidExecutor {
        val trainingConfig = TrainingConfig.fromMap(meta)
        
        // Use dynamic trainer registry instead of hardcoded when block
        val trainer = TrainerRegistry.createTrainer(context, method, modelData, trainingConfig)
        
        return AndroidExecutor(trainer)
    }
} 