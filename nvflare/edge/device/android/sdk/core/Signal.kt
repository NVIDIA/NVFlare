package com.nvidia.nvflare.sdk.core

/**
 * Signal for handling abort/stop operations in the edge SDK.
 */
class Signal(private val parent: Signal? = null) {
    private var _value: Any? = null
    private var _triggerTime: Long? = null
    @Volatile
    private var _triggered = false

    /**
     * Get the value of the signal.
     */
    val value: Any?
        get() = _value

    /**
     * Get the trigger time in milliseconds since epoch.
     */
    val triggerTime: Long?
        get() = _triggerTime

    /**
     * Check if the signal has been triggered.
     * If this signal is not triggered, check the parent signal.
     */
    val triggered: Boolean
        get() {
            if (_triggered) {
                return true
            }
            return parent?.triggered ?: false
        }

    /**
     * Trigger the signal with a value.
     * 
     * @param value The value to set for the signal
     */
    fun trigger(value: Any?) {
        _value = value
        _triggerTime = System.currentTimeMillis()
        _triggered = true
    }

    /**
     * Reset the signal.
     * 
     * @param value Optional value to set when resetting
     */
    fun reset(value: Any? = null) {
        _value = value
        _triggerTime = null
        _triggered = false
    }

    /**
     * Check if the signal has been triggered (alias for triggered property).
     */
    val isTriggered: Boolean
        get() = triggered

    /**
     * Get the trigger data (alias for value property).
     */
    fun getTriggerData(): Any? = value
} 