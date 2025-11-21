package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.core.DXO
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.ContextKey
import com.nvidia.nvflare.sdk.core.Signal
import com.nvidia.nvflare.sdk.core.Executor
import com.nvidia.nvflare.sdk.core.AndroidFlareRunner
import com.nvidia.nvflare.sdk.training.ETTrainer
import com.nvidia.nvflare.sdk.training.TrainingConfig
import com.nvidia.nvflare.sdk.utils.TaskHeaderKey
import kotlinx.coroutines.runBlocking

/**
 * ExecuTorch-specific implementation of Executor.
 * Handles ET training execution with dataset creation and result conversion.
 */
class ETTrainerExecutor(
    private val trainer: ETTrainer,
    private val storedArgs: Map<String, Any> = emptyMap()
) : Executor {
    private val TAG = "ETTrainerExecutor"

    override fun execute(taskData: DXO, ctx: Context, abortSignal: Signal): DXO {
        if (abortSignal.isTriggered) {
            throw RuntimeException("Execution aborted")
        }

        try {
            Log.d(TAG, "Starting training execution")
            Log.d(TAG, "Task data keys: ${taskData.data.keys}")
            
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
        // Merge stored args with task data meta (task data takes precedence)
        // This matches the iOS pattern where job-level config is merged with per-task config
        val finalConfig = storedArgs.toMutableMap()
        
        // Task data meta overrides stored args
        val taskMeta = taskData.meta as? Map<String, Any> ?: emptyMap()
        for ((key, value) in taskMeta) {
            finalConfig[key] = value
        }
        
        // Get job name from runner context
        val runner = ctx[ContextKey.RUNNER] as? AndroidFlareRunner
        val jobName = runner?.jobName ?: ""
        Log.d(TAG, "Job name: '$jobName'")
        
        // Add job name to config data for method determination
        finalConfig[TaskHeaderKey.JOB_NAME] = jobName
        
        return TrainingConfig.fromMap(finalConfig)
    }
}

/**
 * Factory for creating ETTrainerExecutor instances.
 */
object ETTrainerExecutorFactory {
    
    fun createExecutor(context: android.content.Context, method: String, meta: Map<String, Any>): ETTrainerExecutor {
        // Create trainer with meta (used for artifact management)
        val trainer = ETTrainer(context, meta)
        
        // Pass meta as stored args to executor (matches iOS pattern)
        // These stored args will be merged with per-task config, with task config taking precedence
        return ETTrainerExecutor(trainer, meta)
    }
}
