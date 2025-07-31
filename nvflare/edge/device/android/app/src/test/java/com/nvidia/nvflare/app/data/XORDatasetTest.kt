package com.nvidia.nvflare.app.data

import org.junit.Test
import org.junit.Assert.*
import com.nvidia.nvflare.sdk.defs.SimpleBatch

/**
 * Test class for XORDataset to verify it matches iOS functionality.
 */
class XORDatasetTest {
    
    @Test
    fun testXORDatasetSize() {
        val dataset = XORDataset()
        assertEquals("Dataset should have 4 XOR data points", 4, dataset.size())
    }
    
    @Test
    fun testXORDatasetDimensions() {
        val dataset = XORDataset()
        assertEquals("Input dimension should be 2", 2, dataset.inputDim())
        assertEquals("Label dimension should be 1", 1, dataset.labelDim())
    }
    
    @Test
    fun testXORDatasetBatchGeneration() {
        val dataset = XORDataset()
        
        // Get first batch with size 2
        val batch1 = dataset.getNextBatch(2)
        assertNotNull("First batch should not be null", batch1)
        assertTrue("Batch should be SimpleBatch", batch1 is SimpleBatch)
        
        val input1 = batch1.getInput() as FloatArray
        val label1 = batch1.getLabel() as FloatArray
        
        assertEquals("First batch should have 4 input values (2 samples * 2 features)", 4, input1.size)
        assertEquals("First batch should have 2 label values", 2, label1.size)
        
        // Get second batch with size 2
        val batch2 = dataset.getNextBatch(2)
        assertNotNull("Second batch should not be null", batch2)
        val input2 = batch2.getInput() as FloatArray
        val label2 = batch2.getLabel() as FloatArray
        
        assertEquals("Second batch should have 4 input values", 4, input2.size)
        assertEquals("Second batch should have 2 label values", 2, label2.size)
        
        // Third batch should be null (no more data)
        val batch3 = dataset.getNextBatch(2)
        assertNull("Third batch should be null when no more data", batch3)
    }
    
    @Test
    fun testXORDatasetReset() {
        val dataset = XORDataset()
        
        // Get first batch
        val batch1 = dataset.getNextBatch(2)
        assertNotNull("First batch should not be null", batch1)
        val input1 = batch1.getInput() as FloatArray
        
        // Reset dataset
        dataset.reset()
        
        // Get batch again after reset
        val batch2 = dataset.getNextBatch(2)
        assertNotNull("Second batch should not be null", batch2)
        val input2 = batch2.getInput() as FloatArray
        
        // Should get same data after reset
        assertArrayEquals("Data should be same after reset", input1, input2, 0.001f)
    }
    
    @Test
    fun testXORDatasetShuffle() {
        val dataset = XORDataset()
        
        // Get first batch without shuffle
        val batch1 = dataset.getNextBatch(4)
        assertNotNull("First batch should not be null", batch1)
        val input1 = batch1.getInput() as FloatArray
        
        // Reset and enable shuffle
        dataset.setShuffle(true)
        dataset.reset()
        
        // Get batch with shuffle
        val batch2 = dataset.getNextBatch(4)
        assertNotNull("Second batch should not be null", batch2)
        val input2 = batch2.getInput() as FloatArray
        
        // Data should be same but potentially in different order
        assertEquals("Shuffled batch should have same size", input1.size, input2.size)
        
        // Verify all values are present (order might be different)
        val sorted1 = input1.sorted()
        val sorted2 = input2.sorted()
        assertArrayEquals("Shuffled data should contain same values", sorted1.toFloatArray(), sorted2.toFloatArray(), 0.001f)
    }
    
    @Test
    fun testXORDatasetXORTruthTable() {
        val dataset = XORDataset()
        
        // Get all data
        val batch = dataset.getNextBatch(4)
        assertNotNull("Batch should not be null", batch)
        val inputs = batch.getInput() as FloatArray
        val labels = batch.getLabel() as FloatArray
        
        // Verify XOR truth table:
        // (1,1) -> 0, (0,0) -> 0, (1,0) -> 1, (0,1) -> 1
        
        // Check that we have the expected XOR patterns
        val expectedInputs = floatArrayOf(1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f)
        val expectedLabels = floatArrayOf(0.0f, 0.0f, 1.0f, 1.0f)
        
        // Note: Order might be different due to implementation, so we'll check that all values are present
        val sortedInputs = inputs.sorted()
        val sortedExpectedInputs = expectedInputs.sorted()
        assertArrayEquals("Inputs should contain XOR truth table values", sortedExpectedInputs, sortedInputs, 0.001f)
        
        val sortedLabels = labels.sorted()
        val sortedExpectedLabels = expectedLabels.sorted()
        assertArrayEquals("Labels should contain XOR truth table outputs", sortedExpectedLabels, sortedLabels, 0.001f)
    }
}
