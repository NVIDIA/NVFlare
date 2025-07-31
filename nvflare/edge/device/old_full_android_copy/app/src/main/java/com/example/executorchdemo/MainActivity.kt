/*
package com.example.executorchdemo

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    private lateinit var imageClassifier: ImageClassifier
    private val TAG = "MainActivity"
    private var selectedImage: android.graphics.Bitmap? by mutableStateOf(null)
    private var classificationResult: String? by mutableStateOf(null)
    private var isLoading by mutableStateOf(false)

    private val getContent = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            try {
                val inputStream = contentResolver.openInputStream(it)
                selectedImage = android.graphics.BitmapFactory.decodeStream(inputStream)
                inputStream?.close()
            } catch (e: Exception) {
                Log.e(TAG, "Error loading image: ${e.message}", e)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d(TAG, "onCreate started")

        try {
            // Copy model file from assets to app's files directory
            val modelFile = File(filesDir, "image_classifier.pte")
            Log.d(TAG, "Checking model file at: ${modelFile.absolutePath}")
            
            if (!modelFile.exists()) {
                Log.d(TAG, "Model file not found, copying from assets...")
                try {
                    assets.open("models/image_classifier.pte").use { input ->
                        FileOutputStream(modelFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    Log.d(TAG, "Model file copied successfully")
                } catch (e: Exception) {
                    Log.e(TAG, "Error copying model file: ${e.message}", e)
                    e.printStackTrace()
                }
            } else {
                Log.d(TAG, "Model file already exists")
            }

            Log.d(TAG, "Initializing ImageClassifier")
            imageClassifier = ImageClassifier(this)
            Log.d(TAG, "ImageClassifier initialized successfully")

            setContent {
                MaterialTheme {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Button(
                            onClick = { getContent.launch("image/*") },
                            enabled = !isLoading
                        ) {
                            Text("Select Image")
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        selectedImage?.let { bitmap ->
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "Selected image",
                                modifier = Modifier.size(200.dp)
                            )

                            Spacer(modifier = Modifier.height(16.dp))

                            Button(
                                onClick = {
                                    try {
                                        isLoading = true
                                        classificationResult = null
                                        val logits = imageClassifier.classifyImage(bitmap)
                                        
                                        // Apply softmax to convert logits to probabilities
                                        val maxLogit = logits.maxOrNull() ?: 0f
                                        val expLogits = logits.map { kotlin.math.exp(it - maxLogit) }
                                        val sumExp = expLogits.sum()
                                        val probabilities = expLogits.map { it / sumExp }
                                        
                                        // Get top 5 predictions
                                        val topPredictions = probabilities
                                            .mapIndexed { index, prob -> Pair(index, prob) }
                                            .sortedByDescending { it.second }
                                            .take(5)
                                        
                                        // Format the results
                                        val results = topPredictions.joinToString("\n") { (index, prob) ->
                                            "${ImageNetClasses.CLASSES[index]} (${String.format("%.2f", prob * 100)}%)"
                                        }
                                        
                                        classificationResult = results
                                    } catch (e: Exception) {
                                        Log.e(TAG, "Error during classification: ${e.message}", e)
                                        classificationResult = "Error: ${e.message}"
                                    } finally {
                                        isLoading = false
                                    }
                                },
                                enabled = !isLoading
                            ) {
                                if (isLoading) {
                                    Text("Classifying...")
                                } else {
                                    Text("Classify Image")
                                }
                            }
                        }

                        Spacer(modifier = Modifier.height(16.dp))

                        classificationResult?.let { result ->
                            Text(
                                text = result,
                                style = MaterialTheme.typography.bodyLarge
                            )
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate: ${e.message}", e)
            e.printStackTrace()
        }
    }
}
*/
 */
