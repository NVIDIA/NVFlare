package com.nvidia.nvflare.sdk.training

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Tensor
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Manages training artifacts, logging, and model saving for the ETTrainer.
 * Keeps the main trainer logic clean and focused.
 */
class TrainingArtifactManager(
    private val context: android.content.Context,
    private val meta: Map<String, Any>
) {
    private val TAG = "TrainingArtifactManager"
    
    // Directory structure
    private val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    private val artifactsDir = File(context.filesDir, "training_artifacts_$timestamp")
    private val modelsDir = File(artifactsDir, "models")
    private val logsDir = File(artifactsDir, "logs")
    
    init {
        setupDirectories()
    }
    
    /**
     * Get the artifacts directory path for external access.
     */
    fun getArtifactsDirectory(): String = artifactsDir.absolutePath
    
    /**
     * Setup artifact directories and create initial summary.
     */
    private fun setupDirectories() {
        try {
            artifactsDir.mkdirs()
            modelsDir.mkdirs()
            logsDir.mkdirs()
            
            Log.i(TAG, "Training artifacts directory: ${artifactsDir.absolutePath}")
            
            // Create initial summary
            createInitialSummary()
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to setup artifact directories", e)
        }
    }
    
    /**
     * Create initial training summary with configuration.
     */
    private fun createInitialSummary() {
        try {
            val summaryLog = File(logsDir, "training_summary.txt")
            summaryLog.writeText("""
                Training Session Started: $timestamp
                Artifacts Directory: ${artifactsDir.absolutePath}
                Models Directory: ${modelsDir.absolutePath}
                Logs Directory: ${logsDir.absolutePath}
                
                Meta Configuration:
                ${meta.entries.joinToString("\n") { "  ${it.key}: ${it.value}" }}
                
                ========================================
                
            """.trimIndent())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create initial summary", e)
        }
    }
    
    /**
     * Save the initial model received from server.
     */
    fun saveInitialModel(modelData: String) {
        try {
            // Extract model_buffer from JSON if needed
            val actualModelData = if (modelData.startsWith("{")) {
                val jsonObject = com.google.gson.JsonParser.parseString(modelData).asJsonObject
                val modelBuffer = jsonObject.get("model_buffer")?.asString
                    ?: throw RuntimeException("No model_buffer found in JSON")
                modelBuffer
            } else {
                modelData
            }
            
            // Safely decode model data, handling both base64 and raw binary cases
            val decodedModelData = java.util.Base64.getDecoder().decode(actualModelData)
            
            val initialModelFile = File(modelsDir, "initial_model.pte")
            initialModelFile.writeBytes(decodedModelData)
            Log.i(TAG, "Initial model saved to: ${initialModelFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save initial model", e)
        }
    }
    
    /**
     * Save model parameters to JSON file.
     */
    fun saveModelParameters(
        parameters: Map<String, Tensor>, 
        filename: String,
        description: String
    ) {
        try {
            val paramFile = File(modelsDir, filename)
            val paramDict = toTensorDictionary(parameters)
            
            val jsonContent = buildString {
                appendLine("// $description")
                appendLine("// Saved at: ${Date()}")
                appendLine("// Parameters: ${paramDict.size}")
                appendLine()
                appendLine("{")
                paramDict.forEach { (key, tensor) ->
                    appendLine("  \"$key\": {")
                    appendLine("    \"shape\": ${tensor["shape"]},")
                    appendLine("    \"data\": [${(tensor["data"] as List<Float>).take(10).joinToString(", ")}${if ((tensor["data"] as List<Float>).size > 10) "..." else ""}]")
                    appendLine("  },")
                }
                appendLine("}")
            }
            
            paramFile.writeText(jsonContent)
            Log.i(TAG, "Saved $description to: ${paramFile.absolutePath}")
            
            // Update summary
            updateSummary("$description saved to: ${paramFile.absolutePath}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save model parameters: $filename", e)
        }
    }
    
    /**
     * Log training progress to file.
     */
    fun logTrainingProgress(
        epoch: Int, 
        totalEpochs: Int, 
        step: Int, 
        loss: Float, 
        progressPercent: Int,
        method: String
    ) {
        try {
            val progressLog = File(logsDir, "training_progress.txt")
            val timestamp = SimpleDateFormat("HH:mm:ss.SSS", Locale.US).format(Date())
            val logEntry = "[$timestamp] Epoch $epoch/$totalEpochs, Step $step, Loss: $loss, Progress: $progressPercent%, Method: $method\n"
            
            progressLog.appendText(logEntry)
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log training progress", e)
        }
    }
    
    /**
     * Save tensor differences to JSON file.
     */
    fun saveTensorDifferences(
        tensorDiff: Map<String, Map<String, Any>>,
        method: String
    ) {
        try {
            val diffFile = File(modelsDir, "tensor_differences.json")
            val diffContent = buildString {
                appendLine("// Tensor Differences (Final - Initial)")
                appendLine("// Saved at: ${Date()}")
                appendLine("// Method: $method")
                appendLine()
                appendLine("{")
                tensorDiff.forEach { (key, tensor) ->
                    appendLine("  \"$key\": {")
                    appendLine("    \"shape\": ${tensor["shape"]},")
                    appendLine("    \"data\": [${(tensor["data"] as List<Float>).take(10).joinToString(", ")}${if ((tensor["data"] as List<Float>).size > 10) "..." else ""}]")
                    appendLine("  },")
                }
                appendLine("}")
            }
            diffFile.writeText(diffContent)
            Log.i(TAG, "Tensor differences saved to: ${diffFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save tensor differences", e)
        }
    }
    
    /**
     * Save final training summary.
     */
    fun saveTrainingSummary(
        method: String,
        epochs: Int,
        batchSize: Int,
        learningRate: Float,
        totalSteps: Int,
        totalLoss: Float,
        tensorDiff: Map<String, Map<String, Any>>
    ) {
        try {
            val summaryFile = File(logsDir, "training_summary.txt")
            val summary = """
                
                ========================================
                TRAINING COMPLETED
                ========================================
                
                Method: $method
                Epochs: $epochs
                Batch Size: $batchSize
                Learning Rate: $learningRate
                Total Steps: $totalSteps
                Average Loss: ${totalLoss / totalSteps}
                
                Tensor Differences:
                ${tensorDiff.entries.joinToString("\n") { "  ${it.key}: ${it.value["data"]?.let { data -> if (data is List<*>) "size=${data.size}" else "unknown" } ?: "unknown"}" }}
                
                Artifacts Location: ${artifactsDir.absolutePath}
                
                ========================================
                
            """.trimIndent()
            
            summaryFile.appendText(summary)
            Log.i(TAG, "Training summary saved to: ${summaryFile.absolutePath}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save training summary", e)
        }
    }
    
    /**
     * Update the summary file with additional information.
     */
    private fun updateSummary(message: String) {
        try {
            val summaryLog = File(logsDir, "training_summary.txt")
            summaryLog.appendText("$message\n")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to update summary", e)
        }
    }
    
    /**
     * Convert tensor map to dictionary format.
     */
    private fun toTensorDictionary(parameters: Map<String, Tensor>): Map<String, Map<String, Any>> {
        val tensorDict = mutableMapOf<String, Map<String, Any>>()
        
        for ((key, tensor) in parameters) {
            val shape = tensor.shape()
            val data = tensor.getDataAsFloatArray()
            
            val singleTensorDict = mapOf(
                "shape" to shape.toList(),
                "data" to data.toList()
            )
            
            tensorDict[key] = singleTensorDict
        }
        
        return tensorDict
    }
} 