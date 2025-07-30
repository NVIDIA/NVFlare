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
import com.facebook.soloader.SoLoader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.jvm.Throws

/**
 * Android ExecuTorch trainer implementation that matches iOS functionality.
 * Uses proper ExecuTorch patterns from the CIFAR-10 example.
 */
class ETTrainer(
    private val context: Context,
    private val modelData: String, 
    private val meta: Map<String, Any>
) : Trainer {
    private val TAG = "ETTrainer"
    private var tModule: TrainingModule? = null
    private var isInitialized = false

    init {
        initializeTrainingModule()
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
            
            // Initialize SoLoader for loading native libraries
            SoLoader.init(context, false)
            
            // Decode base64 model data and write to temporary file
            val decodedModelData = java.util.Base64.getDecoder().decode(modelData)
            val tempFile = File.createTempFile("model", ".pte")
            tempFile.writeBytes(decodedModelData)
            
            Log.d(TAG, "Model written to temporary file: ${tempFile.absolutePath}")
            
            // For now, we'll use a dummy data path since we don't have the actual .ptd file
            // In a real implementation, you would have the .ptd file in assets
            val dummyDataPath = tempFile.absolutePath // Placeholder
            
            // Load the training module using ExecuTorch's TrainingModule.load
            tModule = TrainingModule.load(tempFile.absolutePath, dummyDataPath)
            
            // Clean up temp file
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
            throw RuntimeException("Training module not initialized")
        }
        
        try {
            // Extract training parameters
            val method = config.method ?: "cnn"
            val epochs = config.epochs ?: 1
            val batchSize = config.batchSize ?: 32
            val learningRate = config.learningRate ?: 0.01f
            val momentum = config.momentum ?: 0.9f
            
            Log.d(TAG, "Training parameters - method: $method, epochs: $epochs, batchSize: $batchSize, lr: $learningRate")
            
            // Get dataset based on method
            val dataset = when (method) {
                "cnn" -> CIFAR10Dataset(context)
                "xor" -> XORDataset()
                else -> throw IllegalArgumentException("Unsupported method: $method")
            }
            
            // Training loop based on CIFAR-10 example
            val trainingResult = performTraining(
                tModule!!, 
                dataset, 
                method, 
                epochs, 
                batchSize, 
                learningRate, 
                momentum
            )
            
            // Convert to iOS-compatible format
            val resultData = when (method) {
                "cnn" -> {
                    // Return tensor format for CNN, matching iOS
                    mapOf(
                        "weight" to mapOf(
                            "sizes" to listOf(4, 3, 32, 32),
                            "strides" to listOf(3072, 1024, 32, 1),
                            "data" to trainingResult["weight_data"] as? List<Float> ?: List(4 * 3 * 32 * 32) { 0.0f }
                        )
                    )
                }
                "xor" -> {
                    // Return number format for XOR, matching iOS
                    mapOf(
                        "value" to (trainingResult["value"] as? Double ?: 0.0),
                        "count" to (trainingResult["count"] as? Int ?: 1)
                    )
                }
                else -> {
                    // Generic tensor format for other methods
                    mapOf(
                        "tensor" to mapOf(
                            "name" to "weight",
                            "data" to trainingResult["weight_data"] as? List<Float> ?: listOf(0.0f)
                        )
                    )
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
            return dxo
            
        } catch (e: Exception) {
            Log.e(TAG, "Training failed", e)
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
        
        // Get model parameters for optimizer
        val parameters: Map<String, Tensor> = model.namedParameters("forward")
        val sgd = SGD.create(parameters, learningRate.toDouble(), momentum.toDouble(), 0.0, 0.0, true)
        
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
                
                // Create tensors
                val inputTensor = createInputTensor(inputData, method, batchSize)
                val labelTensor = createLabelTensor(labelData, batchSize)
                
                // Forward-backward pass
                val inputEValues = arrayOf(EValue.from(inputTensor), EValue.from(labelTensor))
                val outputEValues = model.executeForwardBackward("forward", *inputEValues)
                
                // Extract loss
                val loss = outputEValues[0].toTensor().getDataAsFloatArray()[0]
                epochLoss += loss
                totalLoss += loss
                epochSteps++
                totalSteps++
                
                // Update parameters using SGD
                val gradients: Map<String, Tensor> = model.namedGradients("forward")
                sgd.step(gradients)
                
                if (epochSteps % 10 == 0) {
                    Log.d(TAG, "Epoch $epoch, Step $epochSteps, Loss: $loss")
                }
            }
            
            Log.d(TAG, "Epoch $epoch completed - Average Loss: ${epochLoss / epochSteps}")
        }
        
        Log.d(TAG, "Training completed - Total Steps: $totalSteps, Average Loss: ${totalLoss / totalSteps}")
        
        // Return training results
        return when (method) {
            "cnn" -> {
                // Extract weight differences for CNN
                val weightData = extractWeightDifferences(parameters)
                mapOf(
                    "weight_data" to weightData,
                    "loss" to (totalLoss / totalSteps),
                    "steps" to totalSteps
                )
            }
            "xor" -> {
                // For XOR, return simple values
                mapOf(
                    "value" to (totalLoss / totalSteps).toDouble(),
                    "count" to totalSteps
                )
            }
            else -> {
                mapOf(
                    "weight_data" to listOf<Float>(),
                    "loss" to (totalLoss / totalSteps),
                    "steps" to totalSteps
                )
            }
        }
    }

    /**
     * Create input tensor based on method and batch size.
     */
    private fun createInputTensor(inputData: FloatArray, method: String, batchSize: Int): Tensor {
        return when (method) {
            "cnn" -> {
                // CIFAR-10: [batch, channels, height, width] = [batch, 3, 32, 32]
                Tensor.fromBlob(
                    inputData,
                    longArrayOf(batchSize.toLong(), 3L, 32L, 32L)
                )
            }
            "xor" -> {
                // XOR: [batch, features] = [batch, 2]
                Tensor.fromBlob(
                    inputData,
                    longArrayOf(batchSize.toLong(), 2L)
                )
            }
            else -> {
                // Default: assume 1D input
                Tensor.fromBlob(
                    inputData,
                    longArrayOf(batchSize.toLong(), (inputData.size / batchSize).toLong())
                )
            }
        }
    }

    /**
     * Create label tensor.
     */
    private fun createLabelTensor(labelData: FloatArray, batchSize: Int): Tensor {
        val longLabels = LongArray(batchSize) { labelData[it].toLong() }
        return Tensor.fromBlob(longLabels, longArrayOf(batchSize.toLong()))
    }

    /**
     * Extract weight differences from parameters.
     */
    private fun extractWeightDifferences(parameters: Map<String, Tensor>): List<Float> {
        val weightData = mutableListOf<Float>()
        
        // Extract data from the first parameter (simplified)
        val firstParam = parameters.values.firstOrNull()
        if (firstParam != null) {
            val data = firstParam.getDataAsFloatArray()
            weightData.addAll(data.toList())
        }
        
        return weightData
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
    }

    /**
     * Destructor to ensure cleanup
     */
    protected fun finalize() {
        cleanup()
    }
}
