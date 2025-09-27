package com.nvidia.nvflare.sdk.core

/**
 * Base Batch implementation that matches the Python reference.
 * Concrete batches should extend this class and override the getInput and getLabel methods.
 */
abstract class Batch {
    /**
     * Get the input data for this batch.
     * 
     * @return The input data
     */
    abstract fun getInput(): Any

    /**
     * Get the label data for this batch.
     * 
     * @return The label data
     */
    abstract fun getLabel(): Any
}

/**
 * Simple batch implementation with basic input and label data.
 */
class SimpleBatch(
    private val input: Any,
    private val label: Any
) : Batch() {
    override fun getInput(): Any = input
    override fun getLabel(): Any = label
} 