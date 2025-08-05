package com.nvidia.nvflare.app.training

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.sdk.training.TrainingConfig
import com.nvidia.nvflare.app.training.Trainer
import com.nvidia.nvflare.app.data.CIFAR10Dataset
import com.nvidia.nvflare.app.data.XORDataset
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.TrainingModule
import org.pytorch.executorch.SGD
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlin.jvm.Throws

/**
 * Android ExecuTorch trainer implementation that matches iOS functionality.
 * Uses proper ExecuTorch patterns from the CIFAR-10 example.
 * Enhanced with comprehensive logging and model artifact saving.
 */
class ETTrainer(
    private val context: Context,
    private val modelData: String, 
    private val meta: Map<String, Any>
) : Trainer {
    private val TAG = "ETTrainer"
    private var tModule: TrainingModule? = null
    private var isInitialized = false
    
    // CRITICAL: Strong reference to keep dataset alive during training
    // This prevents the dataset from being deallocated while ExecuTorch still references it
    private var currentDataset: com.nvidia.nvflare.sdk.defs.Dataset? = null
    
    // Enhanced logging and artifact storage
    private val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    private val artifactsDir = File(context.filesDir, "training_artifacts_$timestamp")
    private val modelsDir = File(artifactsDir, "models")
    private val logsDir = File(artifactsDir, "logs")
    
    init {
        setupArtifactDirectories()
    }

    /**
     * Setup directories for storing training artifacts and logs.
     */
    private fun setupArtifactDirectories() {
        try {
            artifactsDir.mkdirs()
            modelsDir.mkdirs()
            logsDir.mkdirs()
            
            Log.i(TAG, "Training artifacts directory: ${artifactsDir.absolutePath}")
            Log.i(TAG, "Models directory: ${modelsDir.absolutePath}")
            Log.i(TAG, "Logs directory: ${logsDir.absolutePath}")
            
            // Create a summary log file
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
            Log.e(TAG, "Failed to setup artifact directories", e)
        }
    }

    /**
     * Save model parameters to file for debugging and analysis.
     */
    private fun saveModelParameters(
        parameters: Map<String, Tensor>, 
        filename: String,
        description: String
    ) {
        try {
            val paramFile = File(modelsDir, filename)
            val paramDict = toTensorDictionary(parameters)
            
            // Save as JSON for easy inspection
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
            
            // Also log to summary
            val summaryLog = File(logsDir, "training_summary.txt")
            summaryLog.appendText("""
                
                $description saved to: ${paramFile.absolutePath}
                Parameters: ${paramDict.size}
                Total values: ${paramDict.values.sumOf { (it["data"] as List<Float>).size }}
                
            """.trimIndent())
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save model parameters: $filename", e)
        }
    }

    /**
     * Save training progress to log file.
     */
    private fun logTrainingProgress(
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
            
            // Also log to Android logcat for real-time monitoring
            Log.d(TAG, "Epoch $epoch/$totalEpochs, Progress $progressPercent%, Step $step, Loss: $loss")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to log training progress", e)
        }
    }

    /**
     * Save final training results and summary.
     */
    private fun saveTrainingSummary(
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
                Session Duration: ${System.currentTimeMillis() - timestamp.hashCode()} ms
                
                ========================================
                
            """.trimIndent()
            
            summaryFile.appendText(summary)
            Log.i(TAG, "Training summary saved to: ${summaryFile.absolutePath}")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save training summary", e)
        }
    }

    /**
     * Extract asset file to internal storage for native access.
     * Based on the CIFAR-10 example pattern.
     */
    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)

        // Create parent directories if they don't exist
        if (!file.parentFile?.exists()!!) {
            file.parentFile?.mkdirs()
        }

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        try {
            context.resources.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            Log.e(TAG, "Error copying asset $assetName: ${e.message}")
            throw e
        }
    }

    /**
     * Initialize the ExecuTorch training module.
     * Based on the CIFAR-10 example pattern.
     */
    private fun initializeTrainingModule() {
        try {
            Log.d(TAG, "Initializing ExecuTorch training module")
            
            // Initialize NativeLoader for loading native libraries
            if (!NativeLoader.isInitialized()) {
                NativeLoader.init(SystemDelegate())
            }
            
            // Decode base64 model data and write to temporary file
            val decodedModelData = java.util.Base64.getDecoder().decode(modelData)
            val tempFile = File.createTempFile("model", ".pte")
            tempFile.writeBytes(decodedModelData)
            
            Log.d(TAG, "Model written to temporary file: ${tempFile.absolutePath}")
            
            // Save the initial model for debugging
            val initialModelFile = File(modelsDir, "initial_model.pte")
            initialModelFile.writeBytes(decodedModelData)
            Log.i(TAG, "Initial model saved to: ${initialModelFile.absolutePath}")
            
            // Load the training module using ExecuTorch's TrainingModule.load
            // For ExecuTorch models, we typically only need the .pte file
            tModule = TrainingModule.load(tempFile.absolutePath, tempFile.absolutePath)
            
            // Clean up temp file after loading
            tempFile.delete()
            
            if (tModule == null) {
                throw RuntimeException("Failed to initialize training module")
            }
            
            isInitialized = true
            Log.d(TAG, "Training module initialized successfully")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize training module", e)
            throw RuntimeException("Failed to initialize ExecuTorch training module: ${e.message}", e)
        }
    }

    override suspend fun train(config: TrainingConfig): Map<String, Any> {
        Log.d(TAG, "Starting ExecuTorch training with method: ${config.method}")
        
        if (!isInitialized || tModule == null) {
            initializeTrainingModule()
        }
        
        try {
            // Extract training parameters
            val method = config.method
            val epochs = config.totalEpochs
            val batchSize = config.batchSize
            val learningRate = config.learningRate
            val momentum = 0.9f  // Default momentum value
            
            Log.d(TAG, "Training parameters - method: $method, epochs: $epochs, batchSize: $batchSize, lr: $learningRate")
            
            // Get dataset based on method and store strong reference
            currentDataset = when (method) {
                "cnn" -> CIFAR10Dataset(context)
                "xor" -> XORDataset()
                else -> throw IllegalArgumentException("Unsupported method: $method")
            }

            // Training loop based on CIFAR-10 example
            val trainingResult = performTraining(
                tModule!!,
                currentDataset!!,
                method,
                epochs,
                batchSize,
                learningRate,
                momentum
            )
            
            // Convert to iOS-compatible format
            val resultData: Map<String, Any> = when (method) {
                "cnn" -> {
                    // Return tensor differences for CNN (mirrors iOS exactly)
                    trainingResult
                }
                "xor" -> {
                    // Return number format for XOR, matching iOS
                    mapOf(
                        "value" to (trainingResult["value"] as? Double ?: 0.0),
                        "count" to (trainingResult["count"] as? Int ?: 1)
                    )
                }
                else -> {
                    // Return tensor differences for other methods (mirrors iOS)
                    trainingResult
                }
            }

            // Get the expected kind from config, default to "number" for backward compatibility
            val expectedKind = config.kind ?: "number"
            Log.d(TAG, "Expected kind from config: $expectedKind")
            
            // Wrap in DXO format with the expected data_kind
            val dxo = mapOf(
                "kind" to expectedKind,
                "data" to resultData,
                "meta" to mapOf(
                    "learning_rate" to learningRate,
                    "batch_size" to batchSize,
                    "method" to method,
                    "epochs" to epochs
                )
            )

            Log.d(TAG, "Training completed successfully, returning DXO with ${resultData.keys.size} keys")
            
            // Release dataset reference after training
            currentDataset = null
            
            return dxo

        } catch (e: Exception) {
            Log.e(TAG, "Training failed", e)
            
            // Release dataset reference on error too
            currentDataset = null
            
            throw RuntimeException("ExecuTorch training failed: ${e.message}", e)
        }
    }

    /**
     * Perform actual training using ExecuTorch.
     * Based on the CIFAR-10 example training loop.
     */
    private fun performTraining(
        model: TrainingModule,
        dataset: com.nvidia.nvflare.sdk.defs.Dataset,
        method: String,
        epochs: Int,
        batchSize: Int,
        learningRate: Float,
        momentum: Float
    ): Map<String, Any> {
        
        Log.d(TAG, "Starting training loop for $epochs epochs")
        
        // Get initial parameters (mirrors iOS implementation)
        val initialParameters: Map<String, Tensor> = model.namedParameters("forward")
        val oldParams = toTensorDictionary(initialParameters)
        Log.d(TAG, "Captured initial parameters with ${oldParams.size} tensors")
        
        // Save initial parameters for debugging
        saveModelParameters(initialParameters, "initial_parameters.json", "Initial Model Parameters")
        
        // Note: We use CIFAR pattern (create per batch) instead of iOS pattern
        // because Java ExecuTorch requires fresh optimizer instances per batch
        var totalLoss = 0.0f
        var totalSteps = 0
        
        for (epoch in 1..epochs) {
            Log.d(TAG, "Starting Epoch $epoch/$epochs")
            
            // Reset dataset for new epoch
            dataset.reset()
            
            var epochLoss = 0.0f
            var epochSteps = 0
            
            // Training loop
            while (true) {
                val batch = dataset.getNextBatch(batchSize)
                if (batch == null) break // End of dataset
                
                // Get input and label data
                val inputData = batch.getInput() as FloatArray
                val labelData = batch.getLabel() as FloatArray
                
                // Create tensors using proper ExecuTorch pattern
                val inputTensor = createInputTensor(inputData, method, batchSize)
                val labelTensor = createLabelTensor(labelData, batchSize)
                
                // Forward-backward pass
                val inputEValues = arrayOf(EValue.from(inputTensor), EValue.from(labelTensor))
                val outputEValues = model.executeForwardBackward("forward", *inputEValues)
                    ?: throw IllegalStateException("Execution module is not loaded.")
                
                // Extract loss and predictions (mirrors CIFAR example exactly)
                val loss = outputEValues[0].toTensor().getDataAsFloatArray()[0]
                val predictions = outputEValues[1].toTensor().getDataAsLongArray()
                
                epochLoss += loss
                totalLoss += loss
                epochSteps++
                totalSteps++
                
                // Update parameters using SGD (mirrors CIFAR example - create per batch)
                val parameters: Map<String, Tensor> = model.namedParameters("forward")
                val sgd = SGD.create(parameters, learningRate.toDouble(), momentum.toDouble(), 0.0, 0.0, true)
                val gradients: Map<String, Tensor> = model.namedGradients("forward")
                sgd.step(gradients)
                
                // Enhanced progress logging with file output
                val samplesProcessed = epochSteps * batchSize
                if (totalSteps % 500 == 0 || (epoch == epochs && epochSteps == 0)) {
                    val progressPercent = (samplesProcessed.toFloat() * 100 / dataset.size()).toInt()
                    logTrainingProgress(epoch, epochs, totalSteps, loss, progressPercent, method)
                }
            }
            
            Log.d(TAG, "Epoch $epoch completed - Average Loss: ${epochLoss / epochSteps}")
        }
        
        Log.d(TAG, "Training completed - Total Steps: $totalSteps, Average Loss: ${totalLoss / totalSteps}")
        
        // Get final parameters and calculate differences (mirrors iOS implementation)
        val finalParameters: Map<String, Tensor> = model.namedParameters("forward")
        val newParams = toTensorDictionary(finalParameters)
        Log.d(TAG, "Captured final parameters with ${newParams.size} tensors")
        
        // Save final parameters for debugging
        saveModelParameters(finalParameters, "final_parameters.json", "Final Model Parameters")
        
        // Calculate tensor differences (new - old)
        val tensorDiff = calculateTensorDifference(oldParams, newParams)
        Log.d(TAG, "Calculated tensor differences with ${tensorDiff.size} tensors")
        
        // Save tensor differences for debugging
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
        
        // Save training summary
        saveTrainingSummary(method, epochs, batchSize, learningRate, totalSteps, totalLoss, tensorDiff)
        
        // Return training results matching iOS format
        return when (method) {
            "cnn" -> {
                // Return tensor differences for CNN (mirrors iOS)
                tensorDiff
            }
            "xor" -> {
                // For XOR, return simple values (maintains backward compatibility)
                mapOf(
                    "value" to (totalLoss / totalSteps).toDouble(),
                    "count" to totalSteps
                )
            }
            else -> {
                // Return tensor differences for other methods
                tensorDiff
            }
        }
    }

    /**
     * Create input tensor based on method and batch size.
     * Uses proper ExecuTorch tensor creation pattern.
     */
    private fun createInputTensor(inputData: FloatArray, method: String, batchSize: Int): Tensor {
        return when (method) {
            "cnn" -> {
                // CIFAR-10: [batch, channels, height, width] = [batch, 3, 32, 32]
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(
                    buffer,
                    longArrayOf(batchSize.toLong(), 3L, 32L, 32L)
                )
            }
            "xor" -> {
                // XOR: [batch, features] = [batch, 2]
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(
                    buffer,
                    longArrayOf(batchSize.toLong(), 2L)
                )
            }
            else -> {
                // Default: assume 1D input
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(
                    buffer,
                    longArrayOf(batchSize.toLong(), (inputData.size / batchSize).toLong())
                )
            }
        }
    }

    /**
     * Create label tensor.
     * Uses proper ExecuTorch tensor creation pattern matching CIFAR example exactly.
     */
    private fun createLabelTensor(labelData: FloatArray, batchSize: Int): Tensor {
        val batchLabelBuffer = LongArray(batchSize) { labelData[it].toLong() }
        return Tensor.fromBlob(batchLabelBuffer, longArrayOf(batchSize.toLong()))
    }

    /**
     * Convert tensor map to dictionary format matching iOS implementation.
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

    /**
     * Calculate tensor differences between old and new parameters.
     * Mirrors iOS calculateTensorDifference implementation.
     */
    private fun calculateTensorDifference(
        oldDict: Map<String, Map<String, Any>>,
        newDict: Map<String, Map<String, Any>>
    ): Map<String, Map<String, Any>> {
        val diffDict = mutableMapOf<String, Map<String, Any>>()
        
        for ((key, oldTensor) in oldDict) {
            val newTensor = newDict[key]
            if (newTensor == null) {
                Log.w(TAG, "Warning: Tensor $key not found in new parameters")
                continue
            }
            
            val oldData = oldTensor["data"] as? List<Float>
            val newData = newTensor["data"] as? List<Float>
            
            if (oldData == null || newData == null) {
                Log.w(TAG, "Warning: Invalid data format for tensor $key")
                continue
            }
            
            if (oldData.size != newData.size) {
                Log.w(TAG, "Warning: Tensor $key size mismatch: old=${oldData.size} new=${newData.size}")
                continue
            }
            
            // Calculate differences: new - old
            val diffData = oldData.zip(newData).map { (oldVal, newVal) -> newVal - oldVal }
            
            val diffTensor = mapOf(
                "shape" to (oldTensor["shape"] ?: emptyList<Long>()),
                "data" to diffData
            )
            
            diffDict[key] = diffTensor
        }
        
        return diffDict
    }

    /**
     * Get the artifacts directory path for external access.
     */
    fun getArtifactsDirectory(): String {
        return artifactsDir.absolutePath
    }

    /**
     * Cleanup resources when the trainer is no longer needed.
     */
    fun cleanup() {
        if (isInitialized) {
            try {
                tModule = null
                Log.d(TAG, "Training module cleaned up successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error during cleanup", e)
            }
            isInitialized = false
        }
        
        // Release dataset reference
        currentDataset = null
        Log.d(TAG, "Dataset reference released")
    }

    /**
     * Destructor to ensure cleanup
     */
    protected fun finalize() {
        cleanup()
    }
}
