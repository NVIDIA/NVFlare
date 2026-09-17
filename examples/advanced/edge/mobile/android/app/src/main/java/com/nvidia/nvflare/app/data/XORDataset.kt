package com.nvidia.nvflare.app.data

import com.nvidia.nvflare.sdk.core.Dataset
import com.nvidia.nvflare.sdk.core.Batch
import com.nvidia.nvflare.sdk.core.SimpleBatch
import android.util.Log

/**
 * Android implementation of XOR dataset that matches the iOS SwiftXORDataset functionality.
 * Provides XOR truth table data for federated learning training.
 */
class XORDataset(private val phase: String = "train") : Dataset {
    private val TAG = "XORDataset"
    
    // XOR truth table: (input1, input2) -> output
    // 0 XOR 0 = 0, 0 XOR 1 = 1, 1 XOR 0 = 1, 1 XOR 1 = 0
    private val xorTable = listOf(
        XORDataPoint(floatArrayOf(1.0f, 1.0f), 0),
        XORDataPoint(floatArrayOf(0.0f, 0.0f), 0),
        XORDataPoint(floatArrayOf(1.0f, 0.0f), 1),
        XORDataPoint(floatArrayOf(0.0f, 1.0f), 1)
    )
    
    private var indices: MutableList<Int>
    private var currentIndex: Int = 0
    private var shouldShuffle: Boolean = false
    
    init {
        Log.d(TAG, "Initializing XOR dataset for phase: $phase")
        this.indices = MutableList(xorTable.size) { it }
        reset()
    }
    
    override fun size(): Int {
        return xorTable.size
    }
    
    override fun getNextBatch(batchSize: Int): Batch? {
        if (currentIndex >= xorTable.size) {
            Log.d(TAG, "No more data available, returning null")
            return null
        }
        
        val endIndex = minOf(currentIndex + batchSize, xorTable.size)
        val actualBatchSize = endIndex - currentIndex
        
        Log.d(TAG, "Getting batch: currentIndex=$currentIndex, endIndex=$endIndex, actualBatchSize=$actualBatchSize")
        
        // Prepare batch data
        val inputs = mutableListOf<Float>()
        val labels = mutableListOf<Float>()
        
        // Extract data from current batch range
        for (i in currentIndex until endIndex) {
            val dataPoint = xorTable[indices[i]]
            inputs.addAll(dataPoint.inputs.toList())
            labels.add(dataPoint.label.toFloat())
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
            indices = MutableList(xorTable.size) { it }
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
        return 2 // XOR has 2 input features
    }
    
    /**
     * Get the label dimension.
     * 
     * @return label dimension
     */
    override fun labelDim(): Int {
        return 1 // XOR has 1 output (binary classification)
    }

    override fun validate() {
        super.validate()
        
        // XOR dataset should have exactly 4 samples
        if (xorTable.size != 4) {
            throw DatasetError.InvalidDataFormat("XOR dataset should have exactly 4 samples, got ${xorTable.size}")
        }
        
        // Validate XOR truth table
        val expectedData = listOf(
            XORDataPoint(floatArrayOf(0f, 0f), 0),
            XORDataPoint(floatArrayOf(0f, 1f), 1),
            XORDataPoint(floatArrayOf(1f, 0f), 1),
            XORDataPoint(floatArrayOf(1f, 1f), 0)
        )
        
        for (i in xorTable.indices) {
            val actual = xorTable[i]
            val expected = expectedData[i]
            
            if (!actual.inputs.contentEquals(expected.inputs) || actual.label != expected.label) {
                throw DatasetError.InvalidDataFormat("XOR dataset contains invalid data at index $i")
            }
        }
    }
    
    /**
     * Data class representing a single XOR data point.
     */
    private data class XORDataPoint(
        val inputs: FloatArray,
        val label: Int
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            
            other as XORDataPoint
            
            if (!inputs.contentEquals(other.inputs)) return false
            if (label != other.label) return false
            
            return true
        }
        
        override fun hashCode(): Int {
            var result = inputs.contentHashCode()
            result = 31 * result + label
            return result
        }
    }
}
