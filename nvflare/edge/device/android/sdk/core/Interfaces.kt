package com.nvidia.nvflare.sdk.core



/**
 * Interface for datasets that provide training data.
 */
interface Dataset {
    fun getNextBatch(batchSize: Int): Batch?
    fun reset()
    fun size(): Int
    fun inputDim(): Int
    fun labelDim(): Int
    fun setShuffle(shuffle: Boolean)
    
    // Add validation method to match iOS SwiftDataset.validate()
    fun validate() {
        // Default implementation - datasets can override for specific validation
        if (size() <= 0) {
            throw DatasetError.EmptyDataset("Dataset size must be greater than 0")
        }
        if (inputDim() <= 0) {
            throw DatasetError.InvalidDataFormat("Input dimension must be greater than 0")
        }
        if (labelDim() <= 0) {
            throw DatasetError.InvalidDataFormat("Label dimension must be greater than 0")
        }
    }
}

/**
 * Interface for data sources that provide datasets.
 */
interface DataSource {
    fun getDataset(datasetType: String, ctx: Context): Dataset
}

/**
 * Interface for executors that perform training tasks.
 */
interface Executor {
    fun execute(taskData: DXO, ctx: Context, abortSignal: Signal): DXO
}

sealed class DatasetError(message: String) : Exception(message) {
    class NoDataFound(message: String) : DatasetError(message)
    class InvalidDataFormat(message: String) : DatasetError(message)
    class EmptyDataset(message: String) : DatasetError(message)
}

 