# Lazy Model Instantiation Tests

This test suite covers the lazy model instantiation feature for PyTorch models in NVFlare.

## Test Structure

### TestPTModelLazyInstantiation
Tests for the `PTModel` class which wraps model configurations:
- `test_ptmodel_with_nn_module`: Verifies existing nn.Module support still works
- `test_ptmodel_with_dict_config`: Tests lazy instantiation with dict config containing path and args
- `test_ptmodel_with_dict_config_no_args`: Tests lazy instantiation with minimal dict config

### TestPTFileModelPersistorLazyInstantiation
Tests for the `PTFileModelPersistor` class which handles model persistence:
- `test_persistor_with_nn_module`: Existing functionality with nn.Module instances
- `test_persistor_with_dict_config`: Lazy instantiation of model from dict config at runtime
- `test_persistor_with_dict_config_no_args`: Lazy instantiation with default model args
- `test_persistor_with_invalid_dict_config_missing_path`: Error handling for missing 'path' key
- `test_persistor_with_invalid_class_path`: Error handling for non-existent model classes
- `test_persistor_with_non_nn_module_class`: Error handling when instantiated class isn't nn.Module
- `test_persistor_string_component_id_still_works`: Backward compatibility with string component IDs

### TestBackwardCompatibility  
Integration tests ensuring backward compatibility:
- `test_nn_module_still_works`: End-to-end test with existing nn.Module workflow

## Running the Tests

```bash
# Run all lazy instantiation tests
python3 -m pytest tests/unit_test/app_opt/pt/test_lazy_model_instantiation.py -v

# Run specific test class
python3 -m pytest tests/unit_test/app_opt/pt/test_lazy_model_instantiation.py::TestPTModelLazyInstantiation -v

# Run specific test
python3 -m pytest tests/unit_test/app_opt/pt/test_lazy_model_instantiation.py::TestPTFileModelPersistorLazyInstantiation::test_persistor_with_dict_config -v
```

## Test Coverage

The tests cover:
1. **Happy path**: Lazy instantiation with valid configurations
2. **Error handling**: Invalid configurations, missing keys, wrong types
3. **Backward compatibility**: Existing code using nn.Module instances
4. **Edge cases**: Missing args, component IDs, various error scenarios

## Implementation Files Tested

- `nvflare/app_opt/pt/job_config/model.py`: PTModel class
- `nvflare/app_opt/pt/file_model_persistor.py`: PTFileModelPersistor class
