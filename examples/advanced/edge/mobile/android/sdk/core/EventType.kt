package com.nvidia.nvflare.sdk.core

/**
 * Standard event types used in the edge SDK.
 */
object EventType {
    const val BEFORE_TRAIN = "before_train"
    const val AFTER_TRAIN = "after_train"
    const val LOSS_GENERATED = "loss_generated"
} 