package com.nvidia.nvflare.sdk.core

import android.util.Log

/**
 * Shared context for passing data between components in the edge SDK.
 * Extends MutableMap to allow storing arbitrary key-value pairs.
 */
class Context : MutableMap<String, Any> {
    private val TAG = "Context"
    private val data = mutableMapOf<String, Any>()

    /**
     * Fire an event to all registered event handlers.
     */
    fun fireEvent(eventType: String, data: Any, abortSignal: Signal) {
        val handlers = get(ContextKey.EVENT_HANDLERS) as? List<EventHandler>
        handlers?.forEach { handler ->
            try {
                handler.handleEvent(eventType, data, this, abortSignal)
            } catch (e: Exception) {
                Log.e(TAG, "Error in event handler for event $eventType", e)
            }
        }
    }

    // MutableMap implementation
    override val size: Int get() = data.size
    override fun isEmpty(): Boolean = data.isEmpty()
    override fun containsKey(key: String): Boolean = data.containsKey(key)
    override fun containsValue(value: Any): Boolean = data.containsValue(value)
    override fun get(key: String): Any? = data[key]
    override fun put(key: String, value: Any): Any? = data.put(key, value)
    override fun remove(key: String): Any? = data.remove(key)
    override fun putAll(from: Map<out String, Any>) = data.putAll(from)
    override fun clear() = data.clear()
    override val keys: MutableSet<String> get() = data.keys
    override val values: MutableCollection<Any> get() = data.values
    override val entries: MutableSet<MutableMap.MutableEntry<String, Any>> get() = data.entries
} 