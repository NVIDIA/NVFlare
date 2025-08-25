package com.nvidia.nvflare.sdk.utils

import com.google.gson.JsonObject
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive

/**
 * Extension function to convert JsonObject to Map<String, Any>
 */
fun JsonObject.asMap(): Map<String, Any> {
    return entrySet().mapNotNull { (key, value) ->
        val convertedValue = when (value) {
            is JsonPrimitive -> when {
                value.isString -> value.asString
                value.isNumber -> value.asNumber
                value.isBoolean -> value.asBoolean
                else -> null
            }
            is JsonObject -> value.asMap()
            is JsonArray -> value.asList()
            else -> null
        }
        if (convertedValue != null) key to convertedValue else null
    }.toMap()
}

/**
 * Extension function to convert JsonArray to List<Any>
 */
fun JsonArray.asList(): List<Any> {
    return mapNotNull { element ->
        when (element) {
            is JsonPrimitive -> when {
                element.isString -> element.asString
                element.isNumber -> element.asNumber
                element.isBoolean -> element.asBoolean
                else -> null
            }
            is JsonObject -> element.asMap()
            is JsonArray -> element.asList()
            else -> null
        }
    }
} 