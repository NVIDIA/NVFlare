package com.nvidia.nvflare.sdk.config

import android.util.Log
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Filter
import com.nvidia.nvflare.sdk.core.EventHandler
import com.nvidia.nvflare.sdk.ETTrainerExecutorFactory

/**
 * Configuration keys used in training configuration.
 */
object ConfigKey {
    const val NAME = "name"
    const val TYPE = "type"
    const val ARGS = "args"
    const val TRAINER = "trainer"
    const val EXECUTORS = "executors"
    const val COMPONENTS = "components"
    const val IN_FILTERS = "in_filters"
    const val OUT_FILTERS = "out_filters"
    const val HANDLERS = "handlers"
}

/**
 * Configuration error exception.
 */
class ConfigError(message: String) : Exception(message)

/**
 * Training configuration containing all components and settings.
 */
data class TrainConfig(
    val objects: Map<String, Any>,
    val inFilters: List<Filter>?,
    val outFilters: List<Filter>?,
    val eventHandlers: List<EventHandler>?,
    val executors: Map<String, Any>
) {
    /**
     * Find an executor for a given task name.
     */
    fun findExecutor(taskName: String): Any? {
        return if (executors.isEmpty()) {
            objects[ConfigKey.TRAINER]
        } else {
            executors[taskName] ?: executors["*"]
        }
    }
}

/**
 * Component resolver that creates objects from configuration specifications.
 */
abstract class ComponentResolver(
    val compType: String,
    val compName: String,
    val compArgs: Map<String, Any>?
) {
    /**
     * Resolve the component spec and create a device-native object.
     */
    abstract fun resolve(): Any?
}

/**
 * Process training configuration and create a TrainConfig object.
 */
fun processTrainConfig(
    context: android.content.Context,
    config: Map<String, Any>,
    resolverRegistry: Map<String, Class<*>>
): TrainConfig {
    val TAG = "ConfigProcessor"
    
    Log.d(TAG, "ConfigProcessor: Processing config: $config")
    Log.d(TAG, "ConfigProcessor: Config keys: ${config.keys}")
    Log.d(TAG, "ConfigProcessor: Looking for key: ${ConfigKey.COMPONENTS}")
    
    val components = config[ConfigKey.COMPONENTS] as? List<Map<String, Any>>
        ?: throw ConfigError("missing ${ConfigKey.COMPONENTS} in config")

    // Process components
    val objTable = processComponents(context, components, resolverRegistry)

    // Process input filters
    val inFilters = config[ConfigKey.IN_FILTERS] as? List<String>
    val processedInFilters = if (inFilters != null) {
        processRefs(inFilters, objTable)
    } else null

    // Process output filters
    val outFilters = config[ConfigKey.OUT_FILTERS] as? List<String>
    val processedOutFilters = if (outFilters != null) {
        processRefs(outFilters, objTable)
    } else null

    // Process event handlers
    val handlers = config[ConfigKey.HANDLERS] as? List<String>
    val processedHandlers = if (handlers != null) {
        processRefs(handlers, objTable)
    } else null

    // Process executors
    val executorConfig = config[ConfigKey.EXECUTORS] as? Map<String, Any>
    val executors = if (executorConfig != null) {
        executorConfig.mapValues { (_, value) ->
            resolveRef(value.toString(), objTable)
        }
    } else {
        emptyMap()
    }

    return TrainConfig(
        objects = objTable,
        inFilters = processedInFilters?.map { it as Filter },
        outFilters = processedOutFilters?.map { it as Filter },
        eventHandlers = processedHandlers?.map { it as EventHandler },
        executors = executors
    )
}

/**
 * Process component specifications and create objects.
 */
private fun processComponents(
    context: android.content.Context,
    components: List<Map<String, Any>>,
    resolverRegistry: Map<String, Class<*>>
): Map<String, Any> {
    val TAG = "ConfigProcessor"
    val resolvers = mutableMapOf<String, ComponentResolver>()
    val objTable = mutableMapOf<String, Any>()

    // Create resolvers for each component
    for (component in components) {
        val name = component[ConfigKey.NAME] as? String
            ?: throw ConfigError("missing name in component")
        
        val compType = component[ConfigKey.TYPE] as? String
            ?: throw ConfigError("missing type in component $name")
        
        val args = component[ConfigKey.ARGS] as? Map<String, Any>
        
        val clazz = resolverRegistry[compType]
            ?: throw ConfigError("no resolver registered for component type $compType")

        val resolver = object : ComponentResolver(compType, name, args) {
            override fun resolve(): Any? {
                return try {
                    // Special handling for executor and trainer types
                    when {
                        compType.startsWith("Executor.") -> {
                            resolveExecutor(args)
                        }
                        compType.startsWith("Trainer.") -> {
                            resolveTrainer(args)
                        }
                        else -> {
                            // Default: create instance using reflection
                            clazz.getDeclaredConstructor().newInstance()
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to create instance of $compType", e)
                    null
                }
            }
            
            private fun resolveExecutor(args: Map<String, Any>?): Any? {
                // Extract required arguments for ETTrainerExecutorFactory
                val method = args?.get("method") as? String ?: "cnn"
                val meta = args?.get("meta") as? Map<String, Any> ?: emptyMap()
                
                // Use ETTrainerExecutorFactory to create the executor
                return ETTrainerExecutorFactory.createExecutor(context, method, meta)
            }
            
            private fun resolveTrainer(args: Map<String, Any>?): Any? {
                // Map Trainer.DLTrainer to Android executor
                // Extract training parameters from args
                val method = args?.get("method") as? String ?: "cnn"
                val epoch = args?.get("epoch") as? Int ?: 5
                val lr = args?.get("lr") as? Double ?: 0.0001
                val optimizer = args?.get("optimizer") as? String ?: "sgd"
                val loss = args?.get("loss") as? String ?: "bce"
                
                // Create meta configuration for the trainer
                val meta = mapOf(
                    "epoch" to epoch,
                    "learning_rate" to lr,
                    "optimizer" to optimizer,
                    "loss" to loss
                )
                
                // Use ETTrainerExecutorFactory with the method from args
                return ETTrainerExecutorFactory.createExecutor(context, method, meta)
            }
        }

        if (name in resolvers) {
            throw ConfigError("duplicate component definition for '$name'")
        }
        resolvers[name] = resolver
    }

    // Resolve all components
    for ((name, resolver) in resolvers) {
        val obj = resolver.resolve()
        if (obj != null) {
            objTable[name] = obj
        }
    }

    return objTable
}

/**
 * Process references to components.
 */
private fun processRefs(refs: List<String>, objTable: Map<String, Any>): List<Any> {
    return refs.map { ref ->
        resolveRef(ref, objTable)
    }
}

/**
 * Resolve a single reference.
 */
private fun resolveRef(ref: String, objTable: Map<String, Any>): Any {
    if (!ref.startsWith("@")) {
        throw ConfigError("invalid reference format: $ref")
    }
    
    val name = ref.substring(1)
    return objTable[name] ?: throw ConfigError("referenced component '$name' does not exist")
} 