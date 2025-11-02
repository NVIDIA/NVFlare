# Installation Notes for NVIDIA FLARE Maintainers

## Adding Feature Election to setup.py

When integrating this module, please add the following to NVFlare's `setup.py`:

### In `extras_require`:
```python
extras_require={
    # ... existing extras ...
    
    "feature_election": [
        "scikit-learn>=1.0.0",
        "PyImpetus>=0.0.6",  # Optional advanced methods
    ],
    
    # Or split into basic/advanced
    "feature_election_basic": [
        "scikit-learn>=1.0.0",
    ],
    
    "feature_election_advanced": [
        "scikit-learn>=1.0.0",
        "PyImpetus>=0.0.6",
    ],
}
```

## User Installation

Then users can install with:
```bash
# Basic (most common)
pip install nvflare[feature_election_basic]

# Advanced (with PyImpetus)
pip install nvflare[feature_election_advanced]

# Or install everything
pip install nvflare[feature_election]
```

## Rationale

- scikit-learn is widely available
- PyImpetus is optional for advanced permutation-based feature selection
- Module works without PyImpetus (gracefully degrades to standard methods)