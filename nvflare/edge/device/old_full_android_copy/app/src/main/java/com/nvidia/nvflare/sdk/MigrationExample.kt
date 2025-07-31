package com.nvidia.nvflare.sdk

import android.content.Context
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.ViewModelStoreOwner
import com.nvidia.nvflare.connection.Connection
import com.nvidia.nvflare.training.TrainerController

/**
 * Example showing how to migrate from the old TrainerController to the new FlareRunnerController.
 * 
 * OLD APPROACH:
 * ```kotlin
 * class MainActivity : AppCompatActivity() {
 *     private lateinit var trainerController: TrainerController
 *     
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *         super.onCreate(savedInstanceState)
 *         
 *         val connection = Connection(this)
 *         trainerController = ViewModelProvider(this)[TrainerController::class.java]
 *         
 *         // Observe training status
 *         trainerController.status.observe(this) { status ->
 *             // Update UI based on status
 *         }
 *         
 *         // Start training
 *         trainerController.startTraining()
 *     }
 * }
 * ```
 * 
 * NEW APPROACH:
 * ```kotlin
 * class MainActivity : AppCompatActivity() {
 *     private lateinit var flareRunnerController: FlareRunnerController
 *     
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *         super.onCreate(savedInstanceState)
 *         
 *         val connection = Connection(this)
 *         flareRunnerController = ViewModelProvider(this)[FlareRunnerController::class.java]
 *         
 *         // Observe training status (same interface!)
 *         flareRunnerController.status.observe(this) { status ->
 *             // Update UI based on status
 *         }
 *         
 *         // Start training (same interface!)
 *         flareRunnerController.startTraining()
 *     }
 * }
 * ```
 */
class MigrationExample {
    
    /**
     * Example of how to create and use the new FlareRunnerController.
     */
    fun createFlareRunnerController(
        context: Context,
        viewModelStoreOwner: ViewModelStoreOwner
    ): FlareRunnerController {
        val connection = Connection(context)
        return ViewModelProvider(viewModelStoreOwner)[FlareRunnerController::class.java]
    }
    
    /**
     * Example of how to set up the connection and start training.
     */
    fun setupAndStartTraining(
        context: Context,
        viewModelStoreOwner: ViewModelStoreOwner,
        hostname: String,
        port: Int
    ) {
        val connection = Connection(context)
        connection.hostname.value = hostname
        connection.port.value = port
        
        val flareRunnerController = ViewModelProvider(viewModelStoreOwner)[FlareRunnerController::class.java]
        
        // The interface is the same as the old TrainerController!
        flareRunnerController.startTraining()
    }
} 