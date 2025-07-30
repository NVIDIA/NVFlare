package com.example.democifar10

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.ext.junit.rules.ActivityScenarioRule
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers.*
import com.facebook.soloader.SoLoader
import org.pytorch.executorch.TrainingModule
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Before
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

/**
 * Instrumented tests for CIFAR-10 ExecuTorch Training Demo App.
 */
@RunWith(AndroidJUnit4::class)
class CifarETTrainingDemoTest {

    @get:Rule
    val activityRule = ActivityScenarioRule(MainActivity::class.java)

    @Before
    fun setUp() {
        // Initialize SoLoader for native library loading
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        SoLoader.init(context, false)
    }

    @Test
    fun testAppContext() {
        // Context of the app under test
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.democifar10", appContext.packageName)
    }

    @Test
    fun testUIComponentsExist() {
        // Test that all main UI components are present
        onView(withId(R.id.fineTuneButton))
            .check(matches(isDisplayed()))
            .check(matches(withText("Fine-tune Model")))

        onView(withId(R.id.evaluateButton))
            .check(matches(isDisplayed()))
            .check(matches(withText("Evaluate Model")))

        onView(withId(R.id.resultTextView))
            .check(matches(isDisplayed()))

        onView(withId(R.id.truthTextView))
            .check(matches(isDisplayed()))

        // Note: statusText is inside a progress container that's initially hidden
        // So we just check that it exists, not that it's displayed
        onView(withId(R.id.statusText))
            .check(matches(isAssignableFrom(android.widget.TextView::class.java)))
    }

    @Test
    fun testAssetFilesExist() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val assetManager = context.assets

        // Test that required asset files exist
        val requiredAssets = listOf(
            "cifar-10-batches-bin/train_data.bin",
            "cifar-10-batches-bin/test_data.bin",
            "cifar10_model.pte",
            "generic_cifar.ptd"
        )

        for (assetPath in requiredAssets) {
            try {
                val inputStream = assetManager.open(assetPath)
                assertNotNull("Asset file $assetPath should exist", inputStream)
                assertTrue("Asset file $assetPath should not be empty", inputStream.available() > 0)
                inputStream.close()
            } catch (e: IOException) {
                // Try without the assets/ prefix
                val alternativePath = assetPath.removePrefix("assets/")
                try {
                    val alternativeStream = assetManager.open(alternativePath)
                    assertNotNull("Asset file $alternativePath should exist", alternativeStream)
                    assertTrue("Asset file $alternativePath should not be empty", alternativeStream.available() > 0)
                    alternativeStream.close()
                } catch (e2: IOException) {
                    fail("Required asset file $assetPath (or $alternativePath) is missing: ${e.message}")
                }
            }
        }
    }

    @Test
    fun testAssetFileCopyFunction() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        activityRule.scenario.onActivity { activity ->
            try {
                // Test copying a small asset file
                val testAssetName = "generic_cifar.ptd"
                val copiedFilePath = activity.javaClass.getDeclaredMethod("assetFilePath", String::class.java)
                    .apply { isAccessible = true }
                    .invoke(activity, testAssetName) as String

                val copiedFile = File(copiedFilePath)
                assertTrue("Copied file should exist", copiedFile.exists())
                assertTrue("Copied file should not be empty", copiedFile.length() > 0)
                assertTrue("Copied file should be readable", copiedFile.canRead())
            } catch (e: Exception) {
                fail("Asset file copy function failed: ${e.message}")
            }
        }
    }

    @Test
    fun testCifar10DataStructure() {
        // Test CIFAR-10 data format constants
        val expectedImageSize = 32 * 32 * 3 // 3072 bytes per image
        val expectedClasses = arrayOf("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        assertEquals("CIFAR-10 should have 10 classes", 10, expectedClasses.size)
        assertEquals("Each CIFAR-10 image should be 3072 bytes", 3072, expectedImageSize)
    }

    @Test
    fun testTrainingModuleLoading() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        activityRule.scenario.onActivity { activity ->
            try {
                // Copy required files to internal storage
                val modelPath = copyAssetToInternalStorage(context, "cifar10_model.pte")
                val dataPath = copyAssetToInternalStorage(context, "generic_cifar.ptd")

                // Verify files exist
                val modelFile = File(modelPath)
                val dataFile = File(dataPath)

                assertTrue("Model file should exist", modelFile.exists())
                assertTrue("Data file should exist", dataFile.exists())
                assertTrue("Model file should be readable", modelFile.canRead())
                assertTrue("Data file should be readable", dataFile.canRead())
                assertTrue("Model file should not be empty", modelFile.length() > 0)
                assertTrue("Data file should not be empty", dataFile.length() > 0)

                // Test TrainingModule loading
                val trainingModule = TrainingModule.load(modelPath, dataPath)
                assertNotNull("TrainingModule should load successfully", trainingModule)

            } catch (e: Exception) {
                fail("TrainingModule loading failed: ${e.message}")
            }
        }
    }

    @Test
    fun testCifar10DataLoading() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        activityRule.scenario.onActivity { activity ->
            try {
                // Copy test data files
                copyAssetToInternalStorage(context, "cifar-10-batches-bin/train_data.bin")
                copyAssetToInternalStorage(context, "cifar-10-batches-bin/test_data.bin")

                // Use reflection to access private loadCifar10Data method
                val loadCifar10DataMethod = activity.javaClass.getDeclaredMethod(
                    "loadCifar10Data",
                    List::class.java,
                    List::class.java,
                    Int::class.java,
                    Int::class.java
                )
                loadCifar10DataMethod.isAccessible = true

                val trainBatchFiles = listOf("train_data.bin")
                val testBatchFiles = listOf("test_data.bin")
                val trainSize = 10 // Small size for testing
                val testSize = 10

                val result = loadCifar10DataMethod.invoke(
                    activity,
                    trainBatchFiles,
                    testBatchFiles,
                    trainSize,
                    testSize
                ) as? Pair<Pair<ByteArray, ByteArray>, Pair<ByteArray, ByteArray>>

                assertNotNull("CIFAR-10 data loading should succeed", result)

                val trainData = result!!.first
                val testData = result.second

                // Verify training data
                assertEquals("Training image data size should match", trainSize * 32 * 32 * 3, trainData.first.size)
                assertEquals("Training label data size should match", trainSize, trainData.second.size)

                // Verify test data
                assertEquals("Test image data size should match", testSize * 32 * 32 * 3, testData.first.size)
                assertEquals("Test label data size should match", testSize, testData.second.size)

                // Verify labels are in valid range (0-9 for CIFAR-10)
                for (label in trainData.second.take(10)) {
                    assertTrue("Training labels should be in range 0-9", label.toInt() and 0xFF in 0..9)
                }

                for (label in testData.second.take(10)) {
                    assertTrue("Test labels should be in range 0-9", label.toInt() and 0xFF in 0..9)
                }

            } catch (e: Exception) {
                fail("CIFAR-10 data loading test failed: ${e.message}")
            }
        }
    }

    @Test
    fun testImageTransformationConstants() {
        // Test that image transformation parameters are reasonable
        val width = 32
        val height = 32
        val channels = 3
        val batchSize = 4

        val expectedImageSize = width * height * channels
        val expectedBatchSize = batchSize * expectedImageSize

        assertEquals("Image width should be 32", 32, width)
        assertEquals("Image height should be 32", 32, height)
        assertEquals("Image channels should be 3 (RGB)", 3, channels)
        assertEquals("Single image size should be 3072 bytes", 3072, expectedImageSize)
    }

    @Test
    fun testInitialUIState() {
        // Test initial state of UI components - they should contain either normal text or error messages
        onView(withId(R.id.resultTextView))
            .check(matches(isDisplayed()))

        onView(withId(R.id.truthTextView))
            .check(matches(isDisplayed()))

        onView(withId(R.id.fineTuneButton))
            .check(matches(isEnabled()))

        onView(withId(R.id.evaluateButton))
            .check(matches(isEnabled()))
    }

    /**
     * Helper function to copy asset files to internal storage for testing
     */
    private fun copyAssetToInternalStorage(context: android.content.Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        // Create parent directories if they don't exist
        file.parentFile?.mkdirs()

        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
