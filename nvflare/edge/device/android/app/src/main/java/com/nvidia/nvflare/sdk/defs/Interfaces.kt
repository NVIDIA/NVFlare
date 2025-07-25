package com.nvidia.nvflare.sdk.defs



/**
 * Interface for datasets that provide training data.
 */
interface Dataset {
    fun size(): Int
    fun getNextBatch(batchSize: Int): Batch
    fun reset()
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

 