package com.example.democifar10

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException

/**
 * Utility class for extracting and saving CIFAR-10 images
 */
class Cifar10ImageExtractor {
    companion object {
        private const val TAG = "Cifar10ImageExtractor"

        // Constants for CIFAR-10 format
        private const val IMAGE_WIDTH = 32
        private const val IMAGE_HEIGHT = 32
        private const val NUM_CHANNELS = 3 // RGB
        private const val LABEL_SIZE = 1
        private const val PIXEL_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * NUM_CHANNELS
        private const val RECORD_SIZE = LABEL_SIZE + PIXEL_SIZE // label + image data

        // Class names for CIFAR-10
        private val CLASS_NAMES = arrayOf(
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        )

        /**
         * Extract images from a CIFAR-10 binary file and save them to the app's files directory
         *
         * @param context Android context to get the files directory
         * @param inputStream FileInputStream for the CIFAR-10 binary file
         * @param numImages Number of images to extract
         * @return Array of file paths to the extracted images
         */
        @JvmStatic
        fun extractAndSaveImages(
            context: Context, inputStream: FileInputStream, numImages: Int
        ): Array<String?> {
            val imagePaths = arrayOfNulls<String>(numImages)

            try {
                val buffer = ByteArray(RECORD_SIZE)

                // Create a directory for the images if it doesn't exist
                val imageDir = File(context.filesDir, "cifar10_images")
                if (!imageDir.exists()) {
                    imageDir.mkdir()
                }

                for (i in 0 until numImages) {
                    // Read one full record (label + image data)
                    val bytesRead = inputStream.read(buffer)
                    if (bytesRead != RECORD_SIZE) {
                        throw IOException("Unexpected end of file or incorrect read size")
                    }

                    // First byte is the label
                    val label = buffer[0].toInt() and 0xFF // Convert to unsigned int
                    val className = CLASS_NAMES[label]

                    // Create a Bitmap to hold the image data
                    val image =
                        Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888)

                    // CIFAR-10 stores images in CHW format (channel, height, width)
                    // Each channel is stored completely before the next channel starts
                    for (y in 0 until IMAGE_HEIGHT) {
                        for (x in 0 until IMAGE_WIDTH) {
                            // Get RGB values
                            val r = buffer[LABEL_SIZE + y * IMAGE_WIDTH + x].toInt() and 0xFF
                            val g =
                                buffer[LABEL_SIZE + IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x].toInt() and 0xFF
                            val b =
                                buffer[LABEL_SIZE + 2 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x].toInt() and 0xFF

                            // Set the pixel color in the bitmap
                            image.setPixel(x, y, Color.rgb(r, g, b))
                        }
                    }

                    // Save the image as PNG
                    val filename = "cifar10_image_${i}_${className}.png"
                    val outputFile = File(imageDir, filename)
                    val outputPath = outputFile.absolutePath

                    FileOutputStream(outputFile).use { out ->
                        image.compress(Bitmap.CompressFormat.PNG, 100, out)
                    }

                    imagePaths[i] = outputPath
                    Log.d(TAG, "Saved image $i with label: $className to $outputPath")
                }
            } catch (e: IOException) {
                Log.e(TAG, "Error extracting images", e)
            }

            return imagePaths
        }

        /**
         * Extract a single image from a CIFAR-10 record (label + image data)
         *
         * @param buffer The buffer containing the CIFAR-10 record
         * @return A Pair containing the label and the bitmap
         */
        @JvmStatic
        fun extractSingleImage(buffer: ByteArray): Pair<Int, Bitmap> {
            // First byte is the label
            val label = buffer[0].toInt() and 0xFF // Convert to unsigned int

            // Create a Bitmap to hold the image data
            val image = Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888)

            // CIFAR-10 stores images in CHW format (channel, height, width)
            // Each channel is stored completely before the next channel starts
            for (y in 0 until IMAGE_HEIGHT) {
                for (x in 0 until IMAGE_WIDTH) {
                    // Get RGB values
                    val r = buffer[LABEL_SIZE + y * IMAGE_WIDTH + x].toInt() and 0xFF
                    val g =
                        buffer[LABEL_SIZE + IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x].toInt() and 0xFF
                    val b =
                        buffer[LABEL_SIZE + 2 * IMAGE_WIDTH * IMAGE_HEIGHT + y * IMAGE_WIDTH + x].toInt() and 0xFF

                    // Set the pixel color in the bitmap
                    image.setPixel(x, y, Color.rgb(r, g, b))
                }
            }

            return Pair(label, image)
        }
    }
}
