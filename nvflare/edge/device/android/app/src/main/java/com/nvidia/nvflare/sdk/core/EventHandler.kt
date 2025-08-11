package com.nvidia.nvflare.sdk.core

/**
 * Base EventHandler implementation that matches the Python reference.
 * Concrete event handlers should extend this class and override the handleEvent method.
 */
abstract class EventHandler {
    /**
     * Handle an event with the given type and data.
     * 
     * @param eventType The type of event (e.g., "before_train", "after_train")
     * @param eventData The data associated with the event
     * @param ctx The context containing shared data
     * @param abortSignal Signal to check for abort requests
     */
    abstract fun handleEvent(eventType: String, eventData: Any, ctx: Context, abortSignal: Signal)
}

/**
 * No-op event handler that does nothing.
 * Useful as a default or placeholder event handler.
 */
class NoOpEventHandler : EventHandler() {
    override fun handleEvent(eventType: String, eventData: Any, ctx: Context, abortSignal: Signal) {
        // Do nothing - this is a no-op implementation
    }
} 