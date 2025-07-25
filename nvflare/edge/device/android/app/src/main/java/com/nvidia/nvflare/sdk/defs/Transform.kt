package com.nvidia.nvflare.sdk.defs

/**
 * Base Transform implementation that matches the Python reference.
 * Concrete transforms should extend this class and override the transform method.
 */
abstract class Transform {
    /**
     * Transform the input batch and return the transformed result.
     * 
     * @param batch The input batch to transform
     * @param ctx The context containing shared data
     * @param abortSignal Signal to check for abort requests
     * @return The transformed batch
     */
    abstract fun transform(batch: Batch, ctx: Context, abortSignal: Signal): Batch
}

/**
 * No-op transform that passes batches through unchanged.
 * Useful as a default or placeholder transform.
 */
class NoOpTransform : Transform() {
    override fun transform(batch: Batch, ctx: Context, abortSignal: Signal): Batch {
        return batch
    }
} 