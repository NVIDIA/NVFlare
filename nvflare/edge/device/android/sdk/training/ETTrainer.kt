package com.nvidia.nvflare.sdk.training

import android.content.Context
import android.util.Log
import com.nvidia.nvflare.sdk.training.TrainingConfig

import com.nvidia.nvflare.sdk.core.Dataset
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.training.TrainingModule
import org.pytorch.executorch.training.SGD
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.jvm.Throws

/**
 * Android ExecuTorch trainer implementation that matches iOS functionality.
 * Implements AutoCloseable for proper resource management.
 * 
 * Usage:
 * ```
 * ETTrainer(context, modelData, meta).use { trainer ->
 *     val dataset = XORDataset("train")  // or CIFAR10Dataset(context)
 *     val result = trainer.train(config, dataset, modelData)
 * }
 * ```
 */
class ETTrainer(
    private val context: android.content.Context,
    private val meta: Map<String, Any>,
    private var dataset: Dataset? = null
) : AutoCloseable {
    private val TAG = "ETTrainer"
    private var tModule: TrainingModule? = null
    
    companion object {
        private const val BUFFER_SIZE = 4 * 1024  // 4KB buffer for file operations
        private const val PROGRESS_LOG_INTERVAL = 500  // Log progress every 500 steps
        private const val DEFAULT_MOMENTUM = 0.9f
        private const val SGD_WEIGHT_DECAY = 0.0
        private const val SGD_NESTEROV = true
        private const val LOSS_TENSOR_INDEX = 0
        private const val PREDICTIONS_TENSOR_INDEX = 1
    }
    

    
    private val artifactManager = TrainingArtifactManager(context, meta)

    init {
        setupArtifactDirectories()
    }
    
    fun setDataset(dataset: Dataset) {
        val previousDataset = this.dataset
        this.dataset = dataset
        
        val previousType = previousDataset?.javaClass?.simpleName
        val currentType = dataset.javaClass.simpleName
        
        if (previousType != null && previousType != currentType) {
            resetModel()
        }
    }
    
    private fun resetModel() {
        tModule = null
    }

    /**
     * Setup artifact directories.
     */
    private fun setupArtifactDirectories() {
        // Artifact setup is now handled by TrainingArtifactManager
        Log.d(TAG, "Artifact directories setup complete")
    }



    /**
     * Extract asset file to internal storage for native access.
     */
    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)

        if (!file.parentFile?.exists()!!) {
            file.parentFile?.mkdirs()
        }

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

                    try {
                context.resources.assets.open(assetName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        val buffer = ByteArray(BUFFER_SIZE)
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
     * 
     * Note: This method handles both base64-encoded and raw binary model data.
     * The NVFlare server should send base64-encoded data, but if it sends raw binary
     * data, this method will attempt to handle it gracefully.
     */
    private fun loadGlobalModel(modelData: String) {
        try {
            Log.d(TAG, "Loading global model from server")
            
            if (!NativeLoader.isInitialized()) {
                NativeLoader.init(SystemDelegate())
            }
            
            // Extract model_buffer from JSON if needed
            val actualModelData = if (modelData.startsWith("{")) {
                // Parse JSON and extract model_buffer
                Log.d(TAG, "Parsing JSON model data")
                val jsonObject = com.google.gson.JsonParser.parseString(modelData).asJsonObject
                Log.d(TAG, "JSON object keys: ${jsonObject.keySet()}")
                val modelBuffer = jsonObject.get("model_buffer")?.asString
                    ?: throw RuntimeException("No model_buffer found in JSON")
                Log.d(TAG, "Extracted model_buffer from JSON, length: ${modelBuffer.length}")
                modelBuffer
            } else {
                // Use as-is if it's already base64
                Log.d(TAG, "Using model data as-is (not JSON)")
                modelData
            }
            
            // Check if the data is already raw binary (not base64)
            Log.d(TAG, "Processing model data, length: ${actualModelData.length}")
            
            val decodedModelData = java.util.Base64.getDecoder().decode(actualModelData)
            
            Log.d(TAG, "Decoded model data size: ${decodedModelData.size} bytes")
            
            // Log first few bytes for debugging (ExecuTorch will validate format internally)
            if (decodedModelData.size >= 8) {
                val headerBytes = decodedModelData.take(8).joinToString(" ") { "%02X".format(it) }
                Log.d(TAG, "Model header bytes (hex): $headerBytes")
            }
            
            val tempFile = File.createTempFile("model", ".pte")
            tempFile.writeBytes(decodedModelData)
            Log.d(TAG, "Written ${decodedModelData.size} bytes to temp file: ${tempFile.absolutePath}")
            
            Log.d(TAG, "Model written to temporary file: ${tempFile.absolutePath}")
            
            // Save initial model for debugging
            artifactManager.saveInitialModel(modelData)
            
            tModule = TrainingModule.load(tempFile.absolutePath)
            tempFile.delete()
            
            if (tModule == null) {
                throw RuntimeException("Failed to initialize training module")
            }
            
            Log.d(TAG, "Global model loaded successfully from server")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load global model from server", e)
            throw RuntimeException("Failed to load global model from server: ${e.message}", e)
        }
    }
    
    suspend fun train(config: TrainingConfig, modelData: String?): Map<String, Any> {
        Log.d(TAG, "Starting ExecuTorch training with method: ${config.method}")
        
        val actualModelData = modelData ?: throw RuntimeException("No model data provided for training")
        
        // Always use the received global model from server
        loadGlobalModel(actualModelData)
        
        try {
            val method = config.method
            val epochs = config.totalEpochs
            val batchSize = config.batchSize
            val learningRate = config.learningRate
            val momentum = DEFAULT_MOMENTUM
            
            Log.d(TAG, "Training parameters - method: $method, epochs: $epochs, batchSize: $batchSize, lr: $learningRate")
            
            // Use dataset provided by user through constructor
            val trainingDataset = dataset ?: throw IllegalStateException("No dataset provided to ETTrainer")
            Log.d(TAG, "Using user-provided dataset: ${trainingDataset.javaClass.simpleName}")
            
            
            val trainingResult = performTraining(
                tModule!!,
                trainingDataset,
                method,
                epochs,
                batchSize,
                learningRate,
                momentum
            )
            
            val resultData: Map<String, Any> = trainingResult

            val expectedKind = config.kind ?: "number"
            Log.d(TAG, "Expected kind from config: $expectedKind")
            
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
     */
    private fun performTraining(
        model: TrainingModule,
        dataset: Dataset,
        method: String,
        epochs: Int,
        batchSize: Int,
        learningRate: Float,
        momentum: Float
    ): Map<String, Any> {
        
        Log.d(TAG, "Starting training loop for $epochs epochs")
        
        // Validate dataset size against batch size (iOS pattern)
        val datasetSize = dataset.size()
        if (datasetSize < batchSize) {
            throw IllegalStateException("Dataset too small for batch size! Dataset size: $datasetSize, Batch size: $batchSize. Need at least $batchSize samples.")
        }
        
        // Calculate batches per epoch (floor division - drop incomplete batches like iOS)
        val numBatchesPerEpoch = datasetSize / batchSize
        val samplesUsedPerEpoch = numBatchesPerEpoch * batchSize
        val droppedSamples = datasetSize - samplesUsedPerEpoch
        
        Log.d(TAG, "Dataset size: $datasetSize, Batch size: $batchSize")
        Log.d(TAG, "Batches per epoch: $numBatchesPerEpoch, Samples used: $samplesUsedPerEpoch, Dropped: $droppedSamples")
        
        if (numBatchesPerEpoch == 0) {
            throw IllegalStateException("Dataset too small for batch size! Dataset size: $datasetSize, Batch size: $batchSize. Need at least $batchSize samples.")
        }
        
        val initialParameters: Map<String, Tensor> = model.namedParameters("forward")
        val oldParams = toTensorDictionary(initialParameters)
        Log.d(TAG, "Captured initial parameters with ${oldParams.size} tensors")
        
        // Save initial parameters
        artifactManager.saveModelParameters(initialParameters, "initial_parameters.json", "Initial Model Parameters")
        
        var totalLoss = 0.0f
        var totalSteps = 0
        
        for (epoch in 1..epochs) {
            Log.d(TAG, "Starting Epoch $epoch/$epochs")
            dataset.reset()
            
            var epochLoss = 0.0f
            var epochSteps = 0
            var epochSamplesProcessed = 0
            
            // Process only complete batches (iOS pattern)
            for (batchIdx in 0 until numBatchesPerEpoch) {
                val batch = dataset.getNextBatch(batchSize)
                if (batch == null) break
                
                val inputData = batch.getInput() as FloatArray
                val labelData = batch.getLabel() as FloatArray
                
                // Ensure fixed batch size - drop incomplete batches (iOS pattern)
                val actualBatchSize = labelData.size
                if (actualBatchSize != batchSize) {
                    Log.w(TAG, "Dropping incomplete batch: expected $batchSize samples, got $actualBatchSize samples")
                    break  // Skip remaining incomplete batches in this epoch
                }
                
                Log.d(TAG, "Processing batch - requested batchSize: $batchSize, actual inputData.size: ${inputData.size}, actual labelData.size: ${labelData.size}")
                
                val inputTensor = createInputTensor(inputData, method, batchSize)
                val labelTensor = createLabelTensor(labelData, batchSize)
                
                val inputEValues = arrayOf(EValue.from(inputTensor), EValue.from(labelTensor))
                val outputEValues = model.executeForwardBackward("forward", *inputEValues)
                    ?: throw IllegalStateException("Execution module is not loaded.")
                
                val loss = outputEValues[LOSS_TENSOR_INDEX].toTensor().getDataAsFloatArray()[0]
                val predictions = outputEValues[PREDICTIONS_TENSOR_INDEX].toTensor().getDataAsLongArray()
                
                epochLoss += loss
                totalLoss += loss
                epochSteps++
                totalSteps++
                
                val parameters: Map<String, Tensor> = model.namedParameters("forward")
                val sgd = SGD.create(parameters, learningRate.toDouble(), momentum.toDouble(), SGD_WEIGHT_DECAY, SGD_WEIGHT_DECAY, SGD_NESTEROV)
                val gradients: Map<String, Tensor> = model.namedGradients("forward")
                sgd.step(gradients)
                
                // Track samples processed (all batches are now fixed size)
                epochSamplesProcessed += batchSize
                if (totalSteps % PROGRESS_LOG_INTERVAL == 0 || (epoch == epochs && batchIdx == numBatchesPerEpoch - 1)) {
                    val progressPercent = (epochSamplesProcessed.toFloat() * 100 / samplesUsedPerEpoch).toInt()
                    artifactManager.logTrainingProgress(epoch, epochs, totalSteps, loss, progressPercent, method)
                    Log.d(TAG, "Epoch $epoch/$epochs, Progress $progressPercent%, Step $totalSteps, Loss: $loss")
                }
            }
            
            Log.d(TAG, "Epoch $epoch completed - Average Loss: ${epochLoss / epochSteps}")
        }
        
        Log.d(TAG, "Training completed - Total Steps: $totalSteps, Average Loss: ${totalLoss / totalSteps}")
        
        val finalParameters: Map<String, Tensor> = model.namedParameters("forward")
        val newParams = toTensorDictionary(finalParameters)
        Log.d(TAG, "Captured final parameters with ${newParams.size} tensors")
        
        // Save final parameters
        artifactManager.saveModelParameters(finalParameters, "final_parameters.json", "Final Model Parameters")
        
        val tensorDiff = calculateTensorDifference(oldParams, newParams)
        Log.d(TAG, "Calculated tensor differences with ${tensorDiff.size} tensors")
        
        // Save tensor differences and summary
        artifactManager.saveTensorDifferences(tensorDiff, method)
        artifactManager.saveTrainingSummary(method, epochs, batchSize, learningRate, totalSteps, totalLoss, tensorDiff)
        
        return when (method) {
            "cnn" -> tensorDiff
            "xor" -> tensorDiff  // Return actual tensor differences for server processing
            else -> tensorDiff
        }
    }

    /**
     * Create input tensor based on method and batch size.
     */
    private fun createInputTensor(inputData: FloatArray, method: String, batchSize: Int): Tensor {
        Log.d(TAG, "Creating input tensor - method: $method, inputData.size: ${inputData.size}, requested batchSize: $batchSize")
        
        return when (method) {
            "cnn" -> {
                // CIFAR-10: [batch, channels, height, width] = [batch, 3, 32, 32]
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(buffer, longArrayOf(batchSize.toLong(), 3L, 32L, 32L))
            }
            "xor" -> {
                // XOR: [batch, features] = [batch, 2]
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(buffer, longArrayOf(batchSize.toLong(), 2L))
            }
            else -> {
                val buffer = Tensor.allocateFloatBuffer(inputData.size)
                buffer.put(inputData)
                buffer.rewind()
                Tensor.fromBlob(buffer, longArrayOf(batchSize.toLong(), (inputData.size / batchSize).toLong()))
            }
        }
    }

    /**
     * Create label tensor.
     */
    private fun createLabelTensor(labelData: FloatArray, batchSize: Int): Tensor {
        Log.d(TAG, "Creating label tensor - labelData.size: ${labelData.size}, requested batchSize: $batchSize")
        val batchLabelBuffer = LongArray(batchSize) { labelData[it].toLong() }
        return Tensor.fromBlob(batchLabelBuffer, longArrayOf(batchSize.toLong()))
    }

    /**
     * Convert tensor map to dictionary format.
     */
    private fun toTensorDictionary(parameters: Map<String, Tensor>): Map<String, Map<String, Any>> {
        val tensorDict = mutableMapOf<String, Map<String, Any>>()
        
        for ((key, tensor) in parameters) {
            val shape = tensor.shape()
            val data = tensor.getDataAsFloatArray()
            val sizes = shape.toList()  // Array of dimensions, not total count
            
            // Note: strides() method is not available in Android ExecuTorch Tensor class
            // iOS version includes: val strides = tensor.strides()
            // val strides = tensor.strides()  // Memory layout information (iOS only)
            
            val singleTensorDict = mapOf(
                "shape" to shape.toList(),
                "sizes" to sizes,  // Now matches iOS: array of dimensions
                // "strides" to strides.toList(),  // Not available in Android ExecuTorch
                "data" to data.toList()
            )
            
            tensorDict[key] = singleTensorDict
        }
        
        return tensorDict
    }

    /**
     * Calculate tensor differences between old and new parameters.
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
            
            val diffData = oldData.zip(newData).map { (oldVal, newVal) -> newVal - oldVal }
            
            val diffTensor = mapOf(
                "shape" to (oldTensor["shape"] ?: emptyList<Long>()),
                "sizes" to (oldTensor["sizes"] ?: emptyList<Long>()),  // Array of dimensions
                // "strides" to (oldTensor["strides"] ?: emptyList<Long>()),  // Not available in Android ExecuTorch
                "data" to diffData
            )
            
            diffDict[key] = diffTensor
        }
        
        return diffDict
    }

    /**
     * Get the artifacts directory path for external access.
     */
    fun getArtifactsDirectory(): String = artifactManager.getArtifactsDirectory()

    /**
     * Cleanup resources when the trainer is no longer needed.
     * This method is called automatically when using try-with-resources or when close() is called.
     */
    override fun close() {
        if (tModule != null) {
            try {
                tModule = null
                Log.d(TAG, "Training module cleaned up successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error during cleanup", e)
            }
        }
        Log.d(TAG, "Training module cleaned up")
    }

}
