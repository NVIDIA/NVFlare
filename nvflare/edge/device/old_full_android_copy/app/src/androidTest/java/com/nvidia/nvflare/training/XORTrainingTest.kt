/*
package com.nvidia.nvflare.training

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class XORTrainingTest {
    private lateinit var context: Context
    private lateinit var trainer: ETTrainerWrapper

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        trainer = ETTrainerWrapper("", JobMeta()) // Empty model and meta for now
        
        // Copy the XOR model to the app's files directory
        val modelFile = File(context.filesDir, "xor.pte")
        context.assets.open("xor.pte").use { input ->
            FileOutputStream(modelFile).use { output ->
                input.copyTo(output)
            }
        }
    }

    @Test
    fun testXORTraining() {
        // Load the model
        trainer.loadModel(context.filesDir.absolutePath + "/xor.pte")
        
        // Load the XOR dataset
        trainer.loadDataset("xor")
        
        // Train for a few epochs
        val numEpochs = 100
        val learningRate = 0.1f
        
        for (epoch in 0 until numEpochs) {
            val weightDiff = trainer.train()
            if (epoch % 10 == 0) {
                println("Epoch $epoch, Weight differences: ${weightDiff.joinToString()}")
            }
        }
    }
} */
