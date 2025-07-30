package com.example.democifar10

import kotlin.random.Random

/**
 * Utility class for image transformations similar to PyTorch's torchvision.transforms
 */
object ImageTransformations {
    private const val TAG = "ImageTransformations"

    /**
     * Apply all transformations to a batch of images for training
     * Includes data augmentation: padding, random crop, horizontal flip, and normalization
     *
     * @param imageData Raw image data in bytes (CHW format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Transformed image data as FloatArray
     */
    fun applyTransformations(
        imageData: ByteArray, width: Int = 32, height: Int = 32, channels: Int = 3
    ): FloatArray {
        // Convert byte array to float array (0-255)
        val floatData = imageData.map { it.toUByte().toFloat() }.toFloatArray()

        // Apply transformations
        val paddedData = addPadding(floatData, width, height, channels, padding = 4)
        val croppedData = randomCrop(
            paddedData, width, height, channels, paddedWidth = width + 8, paddedHeight = height + 8
        )
        val flippedData = randomHorizontalFlip(croppedData, width, height, channels)
        val normalizedData = normalize(
            flippedData,
            mean = floatArrayOf(0.4914f, 0.4822f, 0.4465f),
            std = floatArrayOf(0.2023f, 0.1994f, 0.2010f)
        )

        return normalizedData
    }

    /**
     * Apply test transformations to images with no data augmentation
     * Only converts to float and applies normalization
     *
     * @param imageData Raw image data in bytes (CHW format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Transformed image data as FloatArray
     */
    fun applyTestTransformations(
        imageData: ByteArray, width: Int = 32, height: Int = 32, channels: Int = 3
    ): FloatArray {
        // Convert byte array to float array (0-255)
        val floatData = imageData.map { it.toUByte().toFloat() }.toFloatArray()

        // Apply only normalization only
        val normalizedData = normalize(
            floatData,
            mean = floatArrayOf(0.4914f, 0.4822f, 0.4465f),
            std = floatArrayOf(0.2023f, 0.1994f, 0.2010f)
        )

        return normalizedData
    }

    /**
     * Add padding to an image
     *
     * @param imageData Image data as FloatArray (CHW format)
     * @param width Original image width
     * @param height Original image height
     * @param channels Number of channels
     * @param padding Padding size
     * @return Padded image data as FloatArray
     */
    private fun addPadding(
        imageData: FloatArray, width: Int, height: Int, channels: Int, padding: Int
    ): FloatArray {
        val paddedWidth = width + 2 * padding
        val paddedHeight = height + 2 * padding
        val paddedData = FloatArray(channels * paddedWidth * paddedHeight)

        // Initialize with zeros
        paddedData.fill(0f)

        // Copy the original image to the center of the padded image
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val srcIdx = c * height * width + h * width + w
                    val dstIdx =
                        c * paddedHeight * paddedWidth + (h + padding) * paddedWidth + (w + padding)
                    paddedData[dstIdx] = imageData[srcIdx]
                }
            }
        }

        return paddedData
    }

    /**
     * Take a random crop from a padded image
     *
     * @param paddedData Padded image data as FloatArray (CHW format)
     * @param width Target crop width
     * @param height Target crop height
     * @param channels Number of channels
     * @param paddedWidth Width of the padded image
     * @param paddedHeight Height of the padded image
     * @return Cropped image data as FloatArray
     */
    private fun randomCrop(
        paddedData: FloatArray,
        width: Int,
        height: Int,
        channels: Int,
        paddedWidth: Int,
        paddedHeight: Int
    ): FloatArray {
        val croppedData = FloatArray(channels * width * height)

        // Choose a random top-left corner for the crop
        val topOffset = Random.nextInt(paddedHeight - height + 1)
        val leftOffset = Random.nextInt(paddedWidth - width + 1)

        // Extract the crop
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val srcIdx =
                        c * paddedHeight * paddedWidth + (h + topOffset) * paddedWidth + (w + leftOffset)
                    val dstIdx = c * height * width + h * width + w
                    croppedData[dstIdx] = paddedData[srcIdx]
                }
            }
        }

        return croppedData
    }

    /**
     * Randomly flip an image horizontally with 50% probability
     *
     * @param imageData Image data as FloatArray (CHW format)
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Flipped or original image data as FloatArray
     */
    private fun randomHorizontalFlip(
        imageData: FloatArray, width: Int, height: Int, channels: Int
    ): FloatArray {
        // 50% chance to flip
        if (Random.nextBoolean()) {
            return imageData
        }

        val flippedData = FloatArray(imageData.size)

        // Flip horizontally
        for (c in 0 until channels) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val srcIdx = c * height * width + h * width + w
                    val dstIdx = c * height * width + h * width + (width - 1 - w)
                    flippedData[dstIdx] = imageData[srcIdx]
                }
            }
        }

        return flippedData
    }

    /**
     * Normalize image data with given mean and standard deviation
     *
     * @param imageData Image data as FloatArray (CHW format)
     * @param mean Mean values for each channel
     * @param std Standard deviation values for each channel
     * @return Normalized image data as FloatArray
     */
    private fun normalize(
        imageData: FloatArray, mean: FloatArray, std: FloatArray
    ): FloatArray {
        val normalizedData = FloatArray(imageData.size)
        val channels = mean.size
        val pixelsPerChannel = imageData.size / channels

        // Normalize each channel separately
        for (c in 0 until channels) {
            val channelMean = mean[c]
            val channelStd = std[c]

            for (i in 0 until pixelsPerChannel) {
                val idx = c * pixelsPerChannel + i
                // Convert from [0, 255] to [0, 1] and then normalize
                normalizedData[idx] = (imageData[idx] / 255f - channelMean) / channelStd
            }
        }

        return normalizedData
    }

    /**
     * Apply transformations to a batch of images for training
     *
     * @param batchData Batch of image data in bytes
     * @param batchSize Number of images in the batch
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Transformed batch data as FloatArray
     */
    fun applyBatchTransformations(
        batchData: ByteArray, batchSize: Int, width: Int = 32, height: Int = 32, channels: Int = 3
    ): FloatArray {
        val pixelsPerImage = width * height * channels
        val transformedBatch = FloatArray(batchSize * pixelsPerImage)

        for (i in 0 until batchSize) {
            val imageData = ByteArray(pixelsPerImage)
            System.arraycopy(batchData, i * pixelsPerImage, imageData, 0, pixelsPerImage)

            val transformedImage = applyTransformations(imageData, width, height, channels)
            System.arraycopy(
                transformedImage, 0, transformedBatch, i * pixelsPerImage, pixelsPerImage
            )
        }

        return transformedBatch
    }

    /**
     * Apply test transformations to a batch of images
     *
     * @param batchData Batch of image data in bytes
     * @param batchSize Number of images in the batch
     * @param width Image width
     * @param height Image height
     * @param channels Number of channels
     * @return Transformed batch data as FloatArray
     */
    fun applyBatchTestTransformations(
        batchData: ByteArray, batchSize: Int, width: Int = 32, height: Int = 32, channels: Int = 3
    ): FloatArray {
        val pixelsPerImage = width * height * channels
        val transformedBatch = FloatArray(batchSize * pixelsPerImage)

        for (i in 0 until batchSize) {
            val imageData = ByteArray(pixelsPerImage)
            System.arraycopy(batchData, i * pixelsPerImage, imageData, 0, pixelsPerImage)

            val transformedImage = applyTestTransformations(imageData, width, height, channels)
            System.arraycopy(
                transformedImage, 0, transformedBatch, i * pixelsPerImage, pixelsPerImage
            )
        }

        return transformedBatch
    }
}
