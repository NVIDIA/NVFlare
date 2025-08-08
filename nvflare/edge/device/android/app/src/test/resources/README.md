# Test Models for Android/iOS Federated Learning

This directory contains test configurations and model generation scripts for testing the Android and iOS federated learning implementations.

## Available Test Models

### 1. Deterministic XOR Model (Recommended for Testing)
- **Script**: `create_deterministic_xor_test_model.py`
- **Config**: `proto_test_config_deterministic_xor.json`
- **Model**: `deterministic_xor_test_model.pte`

**Characteristics:**
- **Simple**: 2 input features, 2 output classes
- **Architecture**: Linear(2,4) → Sigmoid → Linear(4,2)
- **Training**: Returns simple numeric results (loss value, count)
- **Deterministic**: Fixed weights, fixed learning rate, reproducible results
- **Perfect for**: Reliable testing, debugging, and development
- **Based on**: Reference ExecuTorch training example

**Deterministic Features:**
- Fixed initial weights (no randomness)
- Fixed learning rate (0.1)
- Fixed batch size (4 samples)
- Fixed epochs (5)
- Fixed XOR data order
- Expected loss: ~0.693147 (ln(2))
- Expected count: 20 (4 samples × 5 epochs)

**Usage:**
```bash
# Navigate to test resources directory
cd nvflare/edge/device/android/app/src/test/resources/

# Generate deterministic XOR model and config
python create_deterministic_xor_test_model.py

# Test with routing proxy
python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_deterministic_xor.json
```

### 2. Regular XOR Model (Development)
- **Script**: `create_xor_test_model.py`
- **Config**: `proto_test_config_xor.json`
- **Model**: `xor_test_model.pte`

**Characteristics:**
- **Simple**: 2 input features, 2 output classes
- **Architecture**: Linear(2,4) → Sigmoid → Linear(4,2)
- **Training**: Returns simple numeric results (loss value, count)
- **Non-deterministic**: Random weights, variable results
- **Perfect for**: Development and exploration
- **Based on**: Reference ExecuTorch training example

**Usage:**
```bash
# Generate XOR model and config
python create_xor_test_model.py

# Test with routing proxy
python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_xor.json
```

### 3. Deterministic CNN Model (Recommended for CNN Testing)
- **Script**: `create_deterministic_cnn_test_model.py`
- **Config**: `proto_test_config_deterministic_cnn.json`
- **Model**: `deterministic_cnn_test_model.pte`

**Characteristics:**
- **Simplified**: 1 conv layer + 1 linear layer
- **Architecture**: Conv2d(3,2,2) → MaxPool2d(2,2) → Linear(450,2)
- **Training**: Returns tensor differences (parameter updates)
- **Deterministic**: Fixed weights, fixed learning rate, reproducible results
- **Perfect for**: Reliable CNN testing
- **Based on**: Simplified CIFAR-10 example

**Deterministic Features:**
- Fixed initial weights (no randomness)
- Fixed learning rate (0.01)
- Fixed batch size (2 samples)
- Fixed epochs (3)
- Small input size (8x8 images)
- Simplified architecture
- Predictable tensor differences

**Usage:**
```bash
# Generate deterministic CNN model and config
python create_deterministic_cnn_test_model.py

# Test with routing proxy
python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_deterministic_cnn.json
```

### 4. Regular CNN Model (Development)
- **Script**: `create_test_model.py`
- **Config**: `proto_test_config_with_model.json`
- **Model**: `test_model.pt`

**Characteristics:**
- **Complex**: CIFAR-10 compatible CNN
- **Architecture**: Conv2d layers → Linear layers → 10 classes
- **Training**: Returns tensor differences (parameter updates)
- **Non-deterministic**: Random weights, variable results
- **Perfect for**: Development and exploration
- **Based on**: CIFAR-10 example

**Usage:**
```bash
# Generate CNN model and config
python create_test_model.py

# Test with routing proxy
python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_with_model.json
```

## Model Transfer Protocol

Both Android and iOS apps expect PyTorch models in the following format:

```json
{
  "task_data": {
    "kind": "model",
    "data": {
      "model_buffer": "base64_encoded_model_data"
    },
    "meta": {
      "learning_rate": 0.0001,
      "batch_size": 32,
      "epochs": 5,
      "method": "xor"  // or "cnn"
    }
  }
}
```

## Key Differences

| Aspect | Deterministic XOR | Regular XOR | Deterministic CNN | Regular CNN |
|--------|------------------|-------------|-------------------|-------------|
| **Complexity** | Simple | Simple | Simplified | Complex |
| **Input Size** | 2 features | 2 features | 8x8x3 images | 32x32x3 images |
| **Output Size** | 2 classes | 2 classes | 2 classes | 10 classes |
| **Training Time** | Fast | Fast | Medium | Slower |
| **Result Format** | Numeric | Numeric | Tensor differences | Tensor differences |
| **Determinism** | ✅ Fixed weights | ❌ Random weights | ✅ Fixed weights | ❌ Random weights |
| **Use Case** | Testing | Development | CNN Testing | Development |

## Recommendations

1. **Start with Deterministic XOR**: Use the deterministic XOR model for reliable testing and debugging
2. **Use Regular XOR for Development**: Use the regular XOR model for development and exploration
3. **Test CNN with Deterministic CNN**: Use the deterministic CNN model for reliable CNN testing
4. **Use Regular CNN for Development**: Use the regular CNN model for development and exploration
5. **Real Models**: For production, use actual models from your federated learning jobs

## File Structure

```
test/resources/
├── create_deterministic_xor_test_model.py      # Deterministic XOR model generator
├── create_xor_test_model.py                    # Regular XOR model generator
├── create_deterministic_cnn_test_model.py      # Deterministic CNN model generator
├── create_test_model.py                        # Regular CNN model generator
├── proto_test_config_deterministic_xor.json    # Deterministic XOR test config
├── proto_test_config_xor.json                  # Regular XOR test configuration
├── proto_test_config_deterministic_cnn.json    # Deterministic CNN test config
├── proto_test_config_with_model.json           # Regular CNN test configuration
├── deterministic_xor_test_model.pte            # Generated deterministic XOR model
├── xor_test_model.pte                          # Generated regular XOR model
├── deterministic_cnn_test_model.pte            # Generated deterministic CNN model
├── test_model.pt                               # Generated regular CNN model
└── README.md                                   # This file
```

## Troubleshooting

- **JSON Errors**: Make sure the config files have valid JSON syntax
- **Model Loading**: Ensure the model is in ExecuTorch (.pte) format
- **Base64 Encoding**: The model must be properly base64 encoded
- **Method Matching**: The `method` field must match what the app expects ("xor" or "cnn")
- **Deterministic Testing**: Use the deterministic XOR model for reliable, reproducible test results
- **Random Results**: If you get different results each time, you're using a non-deterministic model 