package com.nvidia.nvflare.sdk.core

/**
 * Base Filter implementation that matches the Python reference.
 * Concrete filters should extend this class and override the filter method.
 */
abstract class Filter {
    /**
     * Filter the input data and return the filtered result.
     * 
     * @param data The input DXO to filter
     * @param ctx The context containing shared data
     * @param abortSignal Signal to check for abort requests
     * @return The filtered DXO
     */
    abstract fun filter(data: DXO, ctx: Context, abortSignal: Signal): DXO
}

/**
 * No-op filter that passes data through unchanged.
 * Useful as a default or placeholder filter.
 */
class NoOpFilter : Filter() {
    override fun filter(data: DXO, ctx: Context, abortSignal: Signal): DXO {
        return data
    }
} 