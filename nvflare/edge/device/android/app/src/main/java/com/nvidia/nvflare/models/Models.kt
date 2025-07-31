package com.nvidia.nvflare.models

import com.google.gson.JsonObject
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive
import java.util.Base64
import com.google.gson.annotations.SerializedName

// Dataset Types
object DatasetType {
    const val CIFAR10 = "cifar10"
    const val XOR = "xor"
}

// Meta Keys
object MetaKey {
    const val DATASET_TYPE = "dataset_type"
    const val BATCH_SIZE = "batch_size"
    const val LEARNING_RATE = "learning_rate"
    const val TOTAL_EPOCHS = "total_epochs"
    const val DATASET_SHUFFLE = "dataset_shuffle"
}

// Model Exchange Format Constants
object ModelExchangeFormat {
    const val MODEL_BUFFER = "model_buffer"
    const val MODEL_BUFFER_TYPE = "model_buffer_type"
    const val MODEL_BUFFER_NATIVE_FORMAT = "model_buffer_native_format"
    const val MODEL_BUFFER_ENCODING = "model_buffer_encoding"
}

// Model Buffer Types
enum class ModelBufferType {
    EXECUTORCH,
    PYTORCH,
    TENSORFLOW,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelBufferType {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

// Model Native Formats
enum class ModelNativeFormat {
    BINARY,
    JSON,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelNativeFormat {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

// Model Encodings
enum class ModelEncoding {
    BASE64,
    RAW,
    UNKNOWN;

    companion object {
        fun fromString(value: String): ModelEncoding {
            return try {
                valueOf(value.uppercase())
            } catch (e: IllegalArgumentException) {
                UNKNOWN
            }
        }
    }
}

// JSON Value Types
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

// Task Headers
object TaskHeaderKey {
    const val TASK_SEQ = "task_seq"
    const val UPDATE_INTERVAL = "update_interval"
    const val CURRENT_ROUND = "current_round"
    const val NUM_ROUNDS = "num_rounds"
    const val CONTRIBUTION_ROUND = "contribution_round"
}

// Training Task
data class TrainingTask(
    val id: String,
    val name: String,
    val jobId: String,
    val modelData: String,
    val trainingConfig: TrainingConfig,
    val currentRound: Int = 0,
    val numRounds: Int = 1,
    val updateInterval: Float = 1.0f
) 