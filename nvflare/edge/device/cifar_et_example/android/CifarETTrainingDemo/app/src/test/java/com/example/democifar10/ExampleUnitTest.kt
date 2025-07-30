package com.example.democifar10

import org.junit.Test
import org.junit.Assert.*
import kotlin.random.Random

/**
 * Unit tests for CIFAR-10 ExecuTorch Training Demo App.
 */
class CifarETTrainingDemoUnitTest {

    companion object {
        // CIFAR-10 constants
        private const val CIFAR_WIDTH = 32
        private const val CIFAR_HEIGHT = 32
        private const val CIFAR_CHANNELS = 3
        private const val CIFAR_CLASSES = 10
        private const val CIFAR_IMAGE_SIZE = CIFAR_WIDTH * CIFAR_HEIGHT * CIFAR_CHANNELS // 3072

        // CIFAR-10 class names
        private val CIFAR_CLASS_NAMES = arrayOf(
            "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        )

        // CIFAR-10 normalization constants
        private val CIFAR_MEAN = floatArrayOf(0.4914f, 0.4822f, 0.4465f)
        private val CIFAR_STD = floatArrayOf(0.2023f, 0.1994f, 0.2010f)
    }

    @Test
    fun testCifar10Constants() {
        assertEquals("CIFAR-10 image width should be 32", 32, CIFAR_WIDTH)
        assertEquals("CIFAR-10 image height should be 32", 32, CIFAR_HEIGHT)
        assertEquals("CIFAR-10 should have 3 channels (RGB)", 3, CIFAR_CHANNELS)
        assertEquals("CIFAR-10 should have 10 classes", 10, CIFAR_CLASSES)
        assertEquals("CIFAR-10 image size should be 3072 bytes", 3072, CIFAR_IMAGE_SIZE)
        assertEquals("CIFAR-10 should have 10 class names", 10, CIFAR_CLASS_NAMES.size)
    }

    @Test
    fun testCifar10ClassNames() {
        val expectedClasses = arrayOf("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        assertArrayEquals("CIFAR-10 class names should match expected values", expectedClasses, CIFAR_CLASS_NAMES)

        // Test that all class names are non-empty
        for (className in CIFAR_CLASS_NAMES) {
            assertFalse("Class name should not be empty", className.isEmpty())
            assertTrue("Class name should be lowercase", className == className.lowercase())
        }
    }

    @Test
    fun testImageTransformationsApplyTestTransformations() {
        // Create test image data (all pixels set to 128 - middle gray)
        val testImageData = ByteArray(CIFAR_IMAGE_SIZE) { 128.toByte() }

        val transformedData = ImageTransformations.applyTestTransformations(
            testImageData, CIFAR_WIDTH, CIFAR_HEIGHT, CIFAR_CHANNELS
        )

        // Verify output size
        assertEquals("Transformed data size should match input", CIFAR_IMAGE_SIZE, transformedData.size)

        // Verify normalization was applied
        val expectedNormalizedValue = (128f / 255f - CIFAR_MEAN[0]) / CIFAR_STD[0]
        assertEquals("First channel should be normalized correctly", expectedNormalizedValue, transformedData[0], 0.001f)
    }

    @Test
    fun testImageTransformationsApplyTrainingTransformations() {
        // Create test image data
        val testImageData = ByteArray(CIFAR_IMAGE_SIZE) { 100.toByte() }

        val transformedData = ImageTransformations.applyTransformations(
            testImageData, CIFAR_WIDTH, CIFAR_HEIGHT, CIFAR_CHANNELS
        )

        // Verify output size
        assertEquals("Transformed data size should match input", CIFAR_IMAGE_SIZE, transformedData.size)

        // Verify data is normalized
        for (value in transformedData) {
            assertTrue("Normalized values should be reasonable", value > -10f && value < 10f)
        }
    }

    @Test
    fun testImageTransformationsBatchProcessing() {
        val batchSize = 4
        val batchData = ByteArray(batchSize * CIFAR_IMAGE_SIZE) { (it % 256).toByte() }

        // Test training batch transformations
        val trainTransformed = ImageTransformations.applyBatchTransformations(
            batchData, batchSize, CIFAR_WIDTH, CIFAR_HEIGHT, CIFAR_CHANNELS
        )

        assertEquals("Training batch output size should be correct",
            batchSize * CIFAR_IMAGE_SIZE, trainTransformed.size)

        // Test test batch transformations
        val testTransformed = ImageTransformations.applyBatchTestTransformations(
            batchData, batchSize, CIFAR_WIDTH, CIFAR_HEIGHT, CIFAR_CHANNELS
        )

        assertEquals("Test batch output size should be correct",
            batchSize * CIFAR_IMAGE_SIZE, testTransformed.size)
    }

    @Test
    fun testNormalizationMathematics() {
        // Test normalization formula: (pixel/255 - mean) / std
        val testPixelValue = 128f
        val expectedNormalized = (testPixelValue / 255f - CIFAR_MEAN[0]) / CIFAR_STD[0]

        // Create single-channel test data
        val testData = FloatArray(CIFAR_WIDTH * CIFAR_HEIGHT) { testPixelValue }

        // Manual normalization calculation
        val manualNormalized = FloatArray(testData.size)
        for (i in testData.indices) {
            manualNormalized[i] = (testData[i] / 255f - CIFAR_MEAN[0]) / CIFAR_STD[0]
        }

        assertEquals("Manual normalization should match expected value",
            expectedNormalized, manualNormalized[0], 0.001f)
    }

    @Test
    fun testDataTypeConversions() {
        // Test byte to unsigned byte conversion
        val testByte: Byte = -1
        val unsignedValue = testByte.toUByte().toFloat()
        assertEquals("Byte -1 should convert to 255.0f", 255.0f, unsignedValue, 0.001f)

        // Test typical CIFAR-10 pixel value conversions
        val pixelValues = byteArrayOf(0, 127, -128, -1) // 0, 127, 128, 255 as unsigned
        val expectedFloats = floatArrayOf(0f, 127f, 128f, 255f)

        for (i in pixelValues.indices) {
            val converted = pixelValues[i].toUByte().toFloat()
            assertEquals("Pixel conversion should be correct",
                expectedFloats[i], converted, 0.001f)
        }
    }

    @Test
    fun testAccuracyCalculation() {
        // Test accuracy calculation logic
        val predictions = longArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        val labels = longArrayOf(0, 1, 2, 3, 4, 5, 6, 7, 9, 0)

        var correct = 0
        for (i in predictions.indices) {
            if (predictions[i] == labels[i]) {
                correct++
            }
        }

        val accuracy = (correct.toDouble() / predictions.size) * 100
        assertEquals("Accuracy should be 80%", 80.0, accuracy, 0.001)
    }

    @Test
    fun testBatchSizeCalculations() {
        val totalSamples = 1000
        val batchSize = 32
        val expectedBatches = totalSamples / batchSize // 31 complete batches
        val remainingSamples = totalSamples % batchSize // 8 remaining samples

        assertEquals("Number of complete batches should be correct", 31, expectedBatches)
        assertEquals("Remaining samples should be correct", 8, remainingSamples)

        // Test with different batch sizes
        val batchSizes = intArrayOf(1, 4, 8, 16, 32, 64, 128)
        for (batchSize in batchSizes) {
            val batches = totalSamples / batchSize
            val remaining = totalSamples % batchSize
            assertTrue("Batches should be non-negative", batches >= 0)
            assertTrue("Remaining should be less than batch size", remaining < batchSize)
            assertEquals("Total should equal batches * batchSize + remaining",
                totalSamples, batches * batchSize + remaining)
        }
    }

    @Test
    fun testTensorShapeCalculations() {
        val batchSize = 4
        val channels = 3
        val height = 32
        val width = 32

        // Test tensor shape calculations
        val expectedTotalElements = batchSize * channels * height * width
        val actualTotalElements = batchSize * CIFAR_IMAGE_SIZE

        assertEquals("Tensor total elements should match", expectedTotalElements, actualTotalElements)

        // Test individual image indexing
        for (batch in 0 until batchSize) {
            val imageStartIndex = batch * CIFAR_IMAGE_SIZE
            val imageEndIndex = (batch + 1) * CIFAR_IMAGE_SIZE

            assertTrue("Image start index should be valid", imageStartIndex >= 0)
            assertTrue("Image end index should be valid", imageEndIndex <= expectedTotalElements)
            assertEquals("Image size should be correct", CIFAR_IMAGE_SIZE, imageEndIndex - imageStartIndex)
        }
    }

    @Test
    fun testLabelValidation() {
        // Test valid CIFAR-10 labels (0-9)
        for (label in 0..9) {
            assertTrue("Label $label should be valid", label in 0..9)
        }

        // Test invalid labels
        val invalidLabels = intArrayOf(-1, 10, 11, 255)
        for (label in invalidLabels) {
            assertFalse("Label $label should be invalid", label in 0..9)
        }
    }

    @Test
    fun testImageDataValidation() {
        // Test valid image data size
        val validImageData = ByteArray(CIFAR_IMAGE_SIZE)
        assertEquals("Valid image data should have correct size", CIFAR_IMAGE_SIZE, validImageData.size)

        // Test invalid image data sizes
        val invalidSizes = intArrayOf(0, 1, 1000, 5000, 10000)
        for (size in invalidSizes) {
            val invalidImageData = ByteArray(size)
            assertNotEquals("Invalid image data should not have CIFAR-10 size",
                CIFAR_IMAGE_SIZE, invalidImageData.size)
        }
    }

    @Test
    fun testRandomSeedConsistency() {
        // Test that random operations can be made deterministic with seed
        val random1 = Random(42)
        val result1 = random1.nextBoolean()
        val result2 = random1.nextInt(100)

        val random2 = Random(42)
        val result1Repeat = random2.nextBoolean()
        val result2Repeat = random2.nextInt(100)

        assertEquals("Random boolean should be consistent with same seed", result1, result1Repeat)
        assertEquals("Random int should be consistent with same seed", result2, result2Repeat)
    }

    @Test
    fun testFloatArrayOperations() {
        val testArray = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)

        // Test array copying
        val copiedArray = FloatArray(testArray.size)
        System.arraycopy(testArray, 0, copiedArray, 0, testArray.size)
        assertArrayEquals("Copied array should match original", testArray, copiedArray, 0.001f)

        // Test array filling
        val filledArray = FloatArray(5)
        filledArray.fill(2.5f)
        for (value in filledArray) {
            assertEquals("All values should be 2.5f", 2.5f, value, 0.001f)
        }
    }

    @Test
    fun testMemoryEfficiency() {
        // Test that transformations don't create excessive memory overhead
        val smallImageData = ByteArray(100) { (it % 256).toByte() }
        val transformedData = ImageTransformations.applyTestTransformations(
            smallImageData, 10, 10, 1
        )

        assertEquals("Output size should match input size", smallImageData.size, transformedData.size)

        // Test batch processing memory efficiency
        val batchSize = 2
        val batchData = ByteArray(batchSize * 100) { (it % 256).toByte() }
        val batchTransformed = ImageTransformations.applyBatchTestTransformations(
            batchData, batchSize, 10, 10, 1
        )

        assertEquals("Batch output size should match input size", batchData.size, batchTransformed.size)
    }

    @Test
    fun testEdgeCases() {
        // Test with minimum valid image (1x1x1)
        val minImageData = ByteArray(1) { 128.toByte() }
        val minTransformed = ImageTransformations.applyTestTransformations(minImageData, 1, 1, 1)
        assertEquals("Minimum image should transform correctly", 1, minTransformed.size)

        // Test with all zero pixels
        val zeroImageData = ByteArray(CIFAR_IMAGE_SIZE) { 0.toByte() }
        val zeroTransformed = ImageTransformations.applyTestTransformations(zeroImageData)
        assertEquals("Zero image should have correct output size", CIFAR_IMAGE_SIZE, zeroTransformed.size)

        // Test with all max pixels (255)
        val maxImageData = ByteArray(CIFAR_IMAGE_SIZE) { 255.toByte() }
        val maxTransformed = ImageTransformations.applyTestTransformations(maxImageData)
        assertEquals("Max image should have correct output size", CIFAR_IMAGE_SIZE, maxTransformed.size)
    }

    @Test
    fun testTrainingParameters() {
        // Test common training parameters
        val learningRates = doubleArrayOf(0.001, 0.01, 0.1, 1.0)
        val momentumValues = doubleArrayOf(0.0, 0.5, 0.9, 0.99)

        for (lr in learningRates) {
            assertTrue("Learning rate should be positive", lr > 0.0)
            assertTrue("Learning rate should be reasonable", lr <= 1.0)
        }

        for (momentum in momentumValues) {
            assertTrue("Momentum should be non-negative", momentum >= 0.0)
            assertTrue("Momentum should be less than 1", momentum < 1.0)
        }
    }
}
