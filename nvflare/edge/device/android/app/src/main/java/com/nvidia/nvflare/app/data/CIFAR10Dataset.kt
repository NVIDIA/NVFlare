package com.nvidia.nvflare.app.data

import com.nvidia.nvflare.sdk.core.Dataset
import com.nvidia.nvflare.sdk.core.Batch
import com.nvidia.nvflare.sdk.core.SimpleBatch
import android.content.Context
import android.util.Log
import java.io.IOException
import java.io.InputStream

/**
 * Android implementation of CIFAR-10 dataset that matches the iOS SwiftCIFAR10Dataset functionality.
 * Loads CIFAR-10 binary data from Android assets and provides it for federated learning training.
 */
class CIFAR10Dataset(
    private val context: Context,
    private val phase: String = "train"
) : Dataset {
    private val TAG = "CIFAR10Dataset"
    
    // CIFAR-10 constants (matching iOS implementation)
    private val imageWidth = 32
    private val imageHeight = 32
    private val channels = 3
    private val imageSize = 32 * 32 * 3 // 3072 bytes per image
    private val maxImages = 16 // Demo limit (matching iOS)
    
    private val images: List<CIFARImage>
    private var indices: MutableList<Int>
    private var currentIndex: Int = 0
    private var shouldShuffle: Boolean = false
    
    init {
        Log.d(TAG, "Initializing CIFAR-10 dataset for phase: $phase")
        this.images = loadCIFAR10Data()
        this.indices = MutableList(images.size) { it }
        reset()
        Log.d(TAG, "CIFAR-10 dataset initialized with ${images.size} images")
    }
    
    override fun size(): Int {
        return images.size
    }
    
    override fun getNextBatch(batchSize: Int): Batch? {
        if (currentIndex >= images.size) {
            Log.d(TAG, "No more data available, returning null")
            return null
        }
        
        val endIndex = minOf(currentIndex + batchSize, images.size)
        val actualBatchSize = endIndex - currentIndex
        
        Log.d(TAG, "Getting batch: currentIndex=$currentIndex, endIndex=$endIndex, actualBatchSize=$actualBatchSize")
        
        // Prepare batch data
        val inputs = mutableListOf<Float>()
        val labels = mutableListOf<Float>()
        
        // Extract data from current batch range
        for (i in currentIndex until endIndex) {
            val image = images[indices[i]]
            inputs.addAll(image.data.toList())
            labels.add(image.label.toFloat())
        }
        
        currentIndex = endIndex
        
        Log.d(TAG, "Batch data: inputs=${inputs.size} values, labels=${labels.size} values")
        
        return SimpleBatch(
            input = inputs.toFloatArray(),
            label = labels.toFloatArray()
        )
    }
    
    override fun reset() {
        Log.d(TAG, "Resetting dataset, shuffle=$shouldShuffle")
        currentIndex = 0
        
        if (shouldShuffle) {
            indices.shuffle()
            Log.d(TAG, "Shuffled indices: $indices")
        } else {
            indices = MutableList(images.size) { it }
            Log.d(TAG, "Reset indices to original order: $indices")
        }
    }
    
    /**
     * Set whether to shuffle the dataset on reset.
     * 
     * @param shuffle true to enable shuffling, false to disable
     */
    override fun setShuffle(shuffle: Boolean) {
        Log.d(TAG, "Setting shuffle to: $shuffle")
        shouldShuffle = shuffle
        reset()
    }
    
    /**
     * Get the input dimension (number of features per sample).
     * 
     * @return input dimension
     */
    override fun inputDim(): Int {
        return imageSize // 3072 features per image (32x32x3)
    }
    
    /**
     * Get the label dimension.
     * 
     * @return label dimension
     */
    override fun labelDim(): Int {
        return 1 // CIFAR-10 has 1 output (10-class classification)
    }
    
    override fun validate() {
        super.validate()
        
        // Additional CIFAR-10 specific validation
        if (images.isEmpty()) {
            throw DatasetError.NoDataFound("CIFAR-10 data not found in app bundle")
        }
        
        if (images.size < 1) { // CIFAR-10 should have at least 1 sample (demo mode)
            throw DatasetError.InvalidDataFormat("CIFAR-10 dataset size is too small: ${images.size}")
        }
        
        // Validate data format (each sample should have 3072 features + 1 label)
        val expectedFeatures = 3072 // 32x32x3
        
        if (images.isNotEmpty()) {
            val sample = images[0]
            if (sample.data.size != expectedFeatures) {
                throw DatasetError.InvalidDataFormat("CIFAR-10 features dimension mismatch: expected $expectedFeatures, got ${sample.data.size}")
            }
        }
    }
    
    /**
     * Load CIFAR-10 binary data from Android assets.
     * 
     * @return List of CIFAR images
     */
    private fun loadCIFAR10Data(): List<CIFARImage> {
        val images = mutableListOf<CIFARImage>()
        
        try {
            // Load binary data from assets
            val inputStream: InputStream = context.assets.open("data_batch_1.bin")
            val bytes = inputStream.readBytes()
            inputStream.close()
            
            Log.d(TAG, "Loaded ${bytes.size} bytes from data_batch_1.bin")
            
            // Validate data format
            val bytesPerImage = 1 + 3072 // 1 byte label + 3072 bytes image data
            if (bytes.size < bytesPerImage) {
                Log.e(TAG, "Data file too small. Expected at least $bytesPerImage bytes, got ${bytes.size}")
                throw DatasetError.InvalidDataFormat("Data file too small")
            }
            
            val numImages = minOf(bytes.size / bytesPerImage, maxImages) // Demo limit
            
            if (numImages <= 0) {
                Log.e(TAG, "No valid images found in data file")
                throw DatasetError.EmptyDataset("No valid images found")
            }
            
            Log.d(TAG, "Processing $numImages CIFAR-10 images")
            
            // Process each image
            for (i in 0 until numImages) {
                val startIndex = i * bytesPerImage
                val label = bytes[startIndex].toInt()
                
                val imageData = mutableListOf<Float>()
                
                // Convert raw bytes to normalized float values [0,1]
                for (j in 1 until bytesPerImage) {
                    val pixelValue = bytes[startIndex + j].toFloat() / 255.0f
                    imageData.add(pixelValue)
                }
                
                images.add(CIFARImage(label = label, data = imageData.toFloatArray()))
            }
            
            Log.d(TAG, "Successfully loaded ${images.size} CIFAR-10 images from assets")
            
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load CIFAR-10 data from assets", e)
            throw DatasetError.NoDataFound("Failed to load CIFAR-10 data: ${e.message}")
        } catch (e: Exception) {
            Log.e(TAG, "Error processing CIFAR-10 data", e)
            throw DatasetError.InvalidDataFormat("Error processing data: ${e.message}")
        }
        
        return images
    }
    
    /**
     * Data class representing a single CIFAR-10 image.
     */
    private data class CIFARImage(
        val label: Int,
        val data: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as CIFARImage
            
            if (label != other.label) return false
            if (!data.contentEquals(other.data)) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = label
            result = 31 * result + data.contentHashCode()
            return result
        }
    }
}

/**
 * Dataset error types matching iOS implementation.
 */
sealed class DatasetError(message: String) : Exception(message) {
    class NoDataFound(message: String) : DatasetError(message)
    class InvalidDataFormat(message: String) : DatasetError(message)
    class EmptyDataset(message: String) : DatasetError(message)
}
