package com.nvidia.nvflare.sdk.core

/**
 * Standard keys used in the Context for accessing common components and data.
 */
object ContextKey {
    const val RUNNER = "runner"
    const val DATA_SOURCE = "data_source"
    const val EXECUTOR = "executor"
    const val COMPONENTS = "components"
    const val EVENT_HANDLERS = "event_handlers"
    const val TASK_NAME = "task_name"
    const val TASK_ID = "task_id"
    const val TASK_DATA = "task_data"
    const val ANDROID_CONTEXT = "android_context"
    const val DATASET = "dataset"  // Store dataset in context (iOS pattern)
} 