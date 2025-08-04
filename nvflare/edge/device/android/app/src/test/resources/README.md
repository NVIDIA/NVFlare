# Test Models for Android/iOS Federated Learning

This directory contains test configurations and model generation scripts for testing the Android and iOS federated learning implementations.

## Available Test Models

### 1. XOR Model (Recommended for Testing)
- **Script**: `create_xor_test_model.py`
- **Config**: `proto_test_config_xor.json`
- **Model**: `xor_test_model.pte`

**Characteristics:**
- **Simple**: 2 input features, 2 output classes
- **Architecture**: Linear(2,4) → Sigmoid → Linear(4,2)
- **Training**: Returns simple numeric results (loss value, count)
- **Perfect for**: Initial testing, debugging, and development
- **Based on**: Reference ExecuTorch training example

**Usage:**
```bash
# Generate XOR model and config
python create_xor_test_model.py

# Test with routing proxy
python -m nvflare.edge.web.routing_proxy 4321 lcp_map.json rootCA.pem proto_test_config_xor.json
```

### 2. CNN Model (Advanced Testing)
- **Script**: `create_test_model.py`
- **Config**: `proto_test_config_with_model.json`
- **Model**: `test_model.pt`

**Characteristics:**
- **Complex**: CIFAR-10 compatible CNN
- **Architecture**: Conv2d layers → Linear layers → 10 classes
- **Training**: Returns tensor differences (parameter updates)
- **Perfect for**: Full federated learning testing
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

| Aspect | XOR Model | CNN Model |
|--------|-----------|-----------|
| **Complexity** | Simple | Complex |
| **Input Size** | 2 features | 32x32x3 images |
| **Output Size** | 2 classes | 10 classes |
| **Training Time** | Fast | Slower |
| **Result Format** | Numeric | Tensor differences |
| **Use Case** | Development | Production testing |

## Recommendations

1. **Start with XOR**: Use the XOR model for initial development and debugging
2. **Progress to CNN**: Use the CNN model for comprehensive testing
3. **Real Models**: For production, use actual models from your federated learning jobs

## File Structure

```
test/resources/
├── create_xor_test_model.py          # XOR model generator
├── create_test_model.py              # CNN model generator
├── proto_test_config_xor.json        # XOR test configuration
├── proto_test_config_with_model.json # CNN test configuration
├── xor_test_model.pte               # Generated XOR model
├── test_model.pt                    # Generated CNN model
└── README.md                        # This file
```

## Troubleshooting

- **JSON Errors**: Make sure the config files have valid JSON syntax
- **Model Loading**: Ensure the model is in ExecuTorch (.pte) format
- **Base64 Encoding**: The model must be properly base64 encoded
- **Method Matching**: The `method` field must match what the app expects ("xor" or "cnn") 