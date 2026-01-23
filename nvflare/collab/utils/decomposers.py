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

"""Flexible decomposer registration for various ML frameworks.

This module provides a flexible way to register FOBS decomposers for
different ML frameworks (PyTorch, TensorFlow, JAX, etc.) based on
what's available in the environment.

Security Note:
    Unlike the removed FOBS auto-register feature, this module:
    - Only registers from a hardcoded, trusted registry
    - Does NOT scan classpath or external configs
    - Custom decomposers must come from 'nvflare.' namespace by default

Usage:
    from nvflare.collab.utils.decomposers import register_decomposers

    # Register all available decomposers
    register_decomposers()

    # Or register specific frameworks
    register_decomposers(frameworks=["torch", "tensorflow"])
"""

import logging
from typing import Dict, List, Optional, Tuple

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.import_utils import optional_import

logger = logging.getLogger(__name__)

# Registry of framework decomposers
# Format: framework_name -> (package_check, decomposer_module, decomposer_class)
DECOMPOSER_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "torch": (
        "torch",
        "nvflare.app_opt.pt.decomposers",
        "TensorDecomposer",
    ),
    "tensorflow": (
        "tensorflow",
        "nvflare.app_opt.tf.decomposers",
        "TFTensorDecomposer",
    ),
    "numpy": (
        "numpy",
        "nvflare.fuel.utils.fobs.decomposers.numpy_decomposers",
        "NumpyArrayDecomposer",
    ),
}

# Track what's been registered to avoid duplicates
_registered_frameworks: set = set()


def register_decomposer(framework: str) -> bool:
    """Register decomposer for a specific framework.

    Args:
        framework: Framework name (e.g., "torch", "tensorflow", "numpy")

    Returns:
        True if successfully registered, False otherwise
    """
    global _registered_frameworks

    if framework in _registered_frameworks:
        logger.debug(f"Decomposer for {framework} already registered")
        return True

    if framework not in DECOMPOSER_REGISTRY:
        logger.warning(f"Unknown framework: {framework}")
        return False

    package_check, decomposer_module, decomposer_class = DECOMPOSER_REGISTRY[framework]

    # Check if the framework package is available
    _, framework_available = optional_import(package_check)
    if not framework_available:
        logger.debug(f"Framework {framework} not installed, skipping decomposer")
        return False

    # Try to import and register the decomposer
    try:
        decomposer, ok = optional_import(module=decomposer_module, name=decomposer_class)
        if ok:
            fobs.register(decomposer)
            _registered_frameworks.add(framework)
            logger.debug(f"Registered {decomposer_class} for {framework}")
            return True
        else:
            logger.debug(f"Decomposer {decomposer_class} not available for {framework}")
            return False
    except Exception as e:
        logger.warning(f"Failed to register decomposer for {framework}: {e}")
        return False


def register_decomposers(frameworks: Optional[List[str]] = None) -> Dict[str, bool]:
    """Register decomposers for specified frameworks.

    Args:
        frameworks: List of framework names to register decomposers for.
                   If None, attempts to register all known frameworks.

    Returns:
        Dictionary mapping framework name to registration success status
    """
    if frameworks is None:
        frameworks = list(DECOMPOSER_REGISTRY.keys())

    results = {}
    for framework in frameworks:
        results[framework] = register_decomposer(framework)

    return results


def register_available_decomposers() -> List[str]:
    """Register decomposers for all available frameworks.

    This is a convenience function that registers decomposers for all
    frameworks that are installed in the current environment.

    Returns:
        List of successfully registered framework names
    """
    results = register_decomposers()
    return [fw for fw, success in results.items() if success]


# Trusted module namespaces for custom decomposers
# Custom decomposers must come from these namespaces for security
TRUSTED_NAMESPACES = ["nvflare."]


def add_custom_decomposer(
    framework: str,
    package_check: str,
    decomposer_module: str,
    decomposer_class: str,
    allow_untrusted: bool = False,
):
    """Add a custom decomposer to the registry.

    This allows users to register their own decomposers for custom
    data types or frameworks.

    Security Note:
        By default, only decomposers from trusted namespaces (nvflare.*)
        are allowed. Set allow_untrusted=True to bypass this check,
        but only do so for modules you fully trust.

    Args:
        framework: Framework/type name (e.g., "jax", "mxnet")
        package_check: Package to check for availability (e.g., "jax")
        decomposer_module: Full module path to the decomposer
        decomposer_class: Class name of the decomposer
        allow_untrusted: If True, allow decomposers from any namespace.
                        Use with caution - only for trusted modules.

    Example:
        # Trusted FLARE decomposer (allowed by default)
        add_custom_decomposer(
            "jax",
            "jax",
            "nvflare.app_opt.jax.decomposers",
            "JaxArrayDecomposer",
        )

        # External decomposer (requires explicit allow)
        add_custom_decomposer(
            "custom",
            "my_package",
            "my_package.decomposers",
            "CustomDecomposer",
            allow_untrusted=True,  # Explicit acknowledgment
        )
        register_decomposer("custom")

    Raises:
        ValueError: If decomposer_module is not from a trusted namespace
                   and allow_untrusted is False.
    """
    # Security check: verify module is from trusted namespace
    if not allow_untrusted:
        is_trusted = any(decomposer_module.startswith(ns) for ns in TRUSTED_NAMESPACES)
        if not is_trusted:
            raise ValueError(
                f"Decomposer module '{decomposer_module}' is not from a trusted namespace. "
                f"Trusted namespaces: {TRUSTED_NAMESPACES}. "
                f"Set allow_untrusted=True if you trust this module."
            )

    DECOMPOSER_REGISTRY[framework] = (package_check, decomposer_module, decomposer_class)
    logger.debug(f"Added custom decomposer for {framework}: {decomposer_module}")


def is_registered(framework: str) -> bool:
    """Check if a framework's decomposer has been registered.

    Args:
        framework: Framework name

    Returns:
        True if registered, False otherwise
    """
    return framework in _registered_frameworks


def get_registered_frameworks() -> List[str]:
    """Get list of frameworks with registered decomposers.

    Returns:
        List of framework names
    """
    return list(_registered_frameworks)


def reset_registry():
    """Reset the registration state (mainly for testing)."""
    global _registered_frameworks
    _registered_frameworks = set()
