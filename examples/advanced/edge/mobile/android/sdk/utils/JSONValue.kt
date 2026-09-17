package com.nvidia.nvflare.sdk.utils

import com.google.gson.JsonObject
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive

// JSON Value Types - Used for web communication protocol
sealed class JSONValue {
    data class StringValue(val value: String) : JSONValue()
    data class IntValue(val value: Int) : JSONValue()
    data class DoubleValue(val value: Double) : JSONValue()
    data class BooleanValue(val value: Boolean) : JSONValue()
    data class ArrayValue(val value: List<JSONValue>) : JSONValue()
    data class ObjectValue(val value: Map<String, JSONValue>) : JSONValue()
    object NullValue : JSONValue()

    companion object {
        fun fromJsonObject(json: JsonObject): ObjectValue {
            val map = mutableMapOf<String, JSONValue>()
            json.entrySet().forEach { (key, value) ->
                map[key] = fromJsonElement(value)
            }
            return ObjectValue(map)
        }

        fun fromJsonArray(json: JsonArray): ArrayValue {
            val list = mutableListOf<JSONValue>()
            json.forEach { element ->
                list.add(fromJsonElement(element))
            }
            return ArrayValue(list)
        }

        fun fromJsonElement(element: JsonElement): JSONValue {
            return when {
                element.isJsonNull -> NullValue
                element.isJsonPrimitive -> {
                    val primitive = element.asJsonPrimitive
                    when {
                        primitive.isString -> StringValue(primitive.asString)
                        primitive.isNumber -> {
                            val number = primitive.asNumber
                            if (number.toDouble() == number.toInt().toDouble()) {
                                IntValue(number.toInt())
                            } else {
                                DoubleValue(number.toDouble())
                            }
                        }
                        primitive.isBoolean -> BooleanValue(primitive.asBoolean)
                        else -> NullValue
                    }
                }
                element.isJsonObject -> fromJsonObject(element.asJsonObject)
                element.isJsonArray -> fromJsonArray(element.asJsonArray)
                else -> NullValue
            }
        }
    }

    // Convert to Any (matching iOS's jsonObject)
    fun toAny(): Any {
        return when (this) {
            is StringValue -> value
            is IntValue -> value
            is DoubleValue -> value
            is BooleanValue -> value
            is ArrayValue -> value.map { it.toAny() }
            is ObjectValue -> value.mapValues { it.value.toAny() }
            is NullValue -> Unit  // Return Unit instead of null to match Any type
        }
    }
}
