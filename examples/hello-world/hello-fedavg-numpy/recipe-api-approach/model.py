# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    numpy model definition
"""

import numpy as np


class SimpleNumpyModel:
    """
    A simple numpy-based model for demonstration purposes.
    This model represents a basic neural network with weights that can be trained
    using federated learning.
    """
    
    def __init__(self, input_size=3, hidden_size=3, output_size=3):
        """
        Initialize the model with random weights.
        
        Args:
            input_size: Size of input layer
            hidden_size: Size of hidden layer  
            output_size: Size of output layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with the standard starting values for this example
        self.weights = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        
    def get_weights(self):
        """Get the current model weights."""
        return self.weights.copy()
    
    def set_weights(self, weights):
        """Set the model weights."""
        self.weights = weights.copy()
    
    def train_step(self, learning_rate=1.0):
        """
        Perform one training step by adding a delta to the weights.
        This is a simplified training step for demonstration purposes.
        
        Args:
            learning_rate: Learning rate for the training step
            
        Returns:
            The updated weights
        """
        # Simple training: add 1 to each weight (simulating gradient descent)
        self.weights = self.weights + learning_rate
        return self.weights
    
    def evaluate(self):
        """
        Evaluate the model and return metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        # Simple evaluation: return the mean of the weights as accuracy
        accuracy = np.mean(self.weights)
        return {"accuracy": accuracy}
