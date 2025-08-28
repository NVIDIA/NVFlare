package com.nvidia.nvflare.sdk.core

/**
 * Data Exchange Object (DXO) - standard format for exchanging data between components.
 */
data class DXO(
    val dataKind: String,
    val data: Map<String, Any>,
    val meta: Map<String, Any> = emptyMap()
) {
    /**
     * Convert DXO to a map representation for serialization.
     */
    fun toMap(): Map<String, Any> {
        return mapOf(
            "kind" to dataKind,
            "data" to data,
            "meta" to meta
        )
    }

    companion object {
        /**
         * Create DXO from a map representation.
         */
        fun fromMap(encoded: Map<String, Any>): DXO {
            return DXO(
                dataKind = encoded["kind"] as? String ?: "",
                data = encoded["data"] as? Map<String, Any> ?: emptyMap(),
                meta = encoded["meta"] as? Map<String, Any> ?: emptyMap()
            )
        }
    }
} 