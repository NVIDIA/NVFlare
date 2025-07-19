package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.defs.DXO
import com.nvidia.nvflare.sdk.defs.Context
import com.nvidia.nvflare.sdk.defs.Signal
import com.nvidia.nvflare.sdk.defs.Executor
import com.nvidia.nvflare.training.Trainer
import com.nvidia.nvflare.training.ETTrainerWrapper
import com.nvidia.nvflare.sdk.training.TrainingConfig
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
            
            // Extract training configuration from task data
            val trainingConfig = extractTrainingConfig(taskData, ctx)
            
            // Execute training using the old Trainer interface
            val result = runBlocking {
                trainer.train(trainingConfig)
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
    fun createExecutor(method: String, modelData: String, meta: Map<String, Any>): AndroidExecutor {
        val trainingConfig = TrainingConfig.fromMap(meta)
        
        // Create the appropriate trainer based on the method
        val trainer = when (method.lowercase()) {
            "cnn" -> createETTrainer(modelData, trainingConfig)
            "xor" -> createETTrainer(modelData, trainingConfig)
            // Future: "custom_method" -> createCustomTrainer(modelData, trainingConfig)
            else -> throw IllegalArgumentException("Unsupported training method: $method")
        }
        
        return AndroidExecutor(trainer)
    }
    
    private fun createETTrainer(modelData: String, meta: TrainingConfig): Trainer {
        // Use the existing ETTrainerWrapper which wraps the trivial ETTrainer
        return ETTrainerWrapper(modelData, meta)
    }
} 