package com.nvidia.nvflare.sdk

import android.util.Log
import com.nvidia.nvflare.sdk.training.Trainer
import com.nvidia.nvflare.sdk.models.TrainingConfig

/**
 * Registry for trainer implementations.
 * Follows the Python pattern for dynamic component resolution.
 */
object TrainerRegistry {
    private val TAG = "TrainerRegistry"
    private val trainerFactories = mutableMapOf<String, TrainerFactory>()

    /**
     * Factory interface for creating trainer instances.
     */
    interface TrainerFactory {
        fun createTrainer(context: android.content.Context, modelData: String, meta: TrainingConfig): Trainer
    }

    /**
     * Register a trainer factory for a specific method name.
     */
    fun registerTrainer(method: String, factory: TrainerFactory) {
        Log.d(TAG, "Registering trainer for method: $method")
        trainerFactories[method.lowercase()] = factory
    }

    /**
     * Register a trainer using a lambda function for a specific method name.
     */
    fun registerTrainer(method: String, factory: (android.content.Context, String, TrainingConfig) -> Trainer) {
        Log.d(TAG, "Registering trainer for method: $method")
        trainerFactories[method.lowercase()] = object : TrainerFactory {
            override fun createTrainer(context: android.content.Context, modelData: String, meta: TrainingConfig): Trainer {
                return factory(context, modelData, meta)
            }
        }
    }

    /**
     * Create a trainer instance for the specified method.
     */
    fun createTrainer(context: android.content.Context, method: String, modelData: String, meta: TrainingConfig): Trainer {
        val factory = trainerFactories[method.lowercase()]
        if (factory == null) {
            val availableMethods = trainerFactories.keys.joinToString(", ")
            throw IllegalArgumentException(
                "No trainer registered for method '$method'. " +
                "Available methods: $availableMethods"
            )
        }
        
        Log.d(TAG, "Creating trainer for method: $method")
        return factory.createTrainer(context, modelData, meta)
    }

    /**
     * Get all registered method names.
     */
    fun getRegisteredMethods(): Set<String> {
        return trainerFactories.keys.toSet()
    }

    /**
     * Check if a method is registered.
     */
    fun isMethodRegistered(method: String): Boolean {
        return trainerFactories.containsKey(method.lowercase())
    }
} 