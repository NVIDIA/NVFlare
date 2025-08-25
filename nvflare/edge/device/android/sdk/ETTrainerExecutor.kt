package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.ContextKey
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.Executor
import com.nvidia.nvflare.sdk.training.ETTrainer
import com.nvidia.nvflare.sdk.training.TrainingConfig
import kotlinx.coroutines.runBlocking

/**
 * ExecuTorch-specific implementation of Executor.
 * Handles ET training execution with dataset creation and result conversion.
 */
class ETTrainerExecutor(
    private val trainer: ETTrainer
) : Executor {
    private val TAG = "ETTrainerExecutor"

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
            
            // Validate model data
            if (modelData.isNullOrEmpty()) {
                Log.e(TAG, "No model data found in task data")
                throw RuntimeException("No model data found in task data")
            }
            
            // Get dataset from context (iOS pattern)
            val dataset = ctx[ContextKey.DATASET] as? com.nvidia.nvflare.sdk.core.Dataset
                ?: throw RuntimeException("No dataset found in context")
            
            Log.d(TAG, "Retrieved dataset from context: ${dataset.javaClass.simpleName}, size: ${dataset.size()}")
            
            // Set dataset on trainer
            trainer.setDataset(dataset)
            
            // Execute training using ETTrainer directly
            val result = runBlocking {
                trainer.train(trainingConfig, modelData)
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
 * Factory for creating ETTrainerExecutor instances.
 */
object ETTrainerExecutorFactory {
    
    /**
     * Create an ETTrainerExecutor based on the training method and model data.
     */
    fun createExecutor(context: android.content.Context, method: String, modelData: String, meta: Map<String, Any>): ETTrainerExecutor {
        val trainingConfig = TrainingConfig.fromMap(meta)

        // Create ETTrainer without dataset - it will be provided by the executor from context
        val trainer = ETTrainer(context, modelData, meta)
        
        return ETTrainerExecutor(trainer)
    }
}
