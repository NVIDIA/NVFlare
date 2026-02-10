.. _differential_privacy:

##############################
Differential Privacy in FLARE
##############################

Overview
========

Differential Privacy (DP) provides mathematically rigorous privacy guarantees for federated learning.
FLARE supports DP at two levels:

- **Local DP (client-side)** -- Privacy filters applied to model updates before sending to the server
- **Sample-level DP (training-time)** -- DP-SGD integration via `Opacus <https://opacus.ai>`_ for per-sample gradient clipping and noise injection during training

Both approaches can be combined with other FLARE privacy mechanisms (homomorphic encryption, secure aggregation)
for defense-in-depth.

DP-SGD with Opacus (Sample-Level DP)
=====================================

For the strongest per-sample privacy guarantees, use DP-SGD during local training. The
:doc:`Hello Differential Privacy </hello-world/hello-dp/index>` example demonstrates a complete
federated DP workflow:

- **Gradient Clipping** -- Per-sample gradients are clipped to bound sensitivity
- **Noise Addition** -- Calibrated Gaussian noise is added to clipped gradients
- **Privacy Accounting** -- Privacy budget (epsilon, delta) is tracked across rounds

The privacy-utility trade-off is controlled by epsilon:

- **Lower epsilon** = stronger privacy, more noise, lower accuracy
- **Higher epsilon** = weaker privacy, less noise, higher accuracy

See the full walkthrough: :doc:`Hello Differential Privacy </hello-world/hello-dp/index>`

Privacy-Preserving Filters (Model Update DP)
=============================================

FLARE's :ref:`filter mechanism <filters>` lets you apply privacy transformations to model updates
before they leave the client. These filters are configured in the job definition and run automatically.

**Built-in privacy filters** (in ``nvflare.app_common.filters``):

``PercentilePrivacy``
    Implements the "largest percentile to share" policy from
    `Shokri & Shmatikov (CCS '15) <https://dl.acm.org/doi/10.1145/2810103.2813687>`_.
    Only weight differences above a configurable percentile are shared; smaller values are zeroed out.

    Parameters: ``percentile`` (default 10), ``gamma`` (clipping threshold, default 0.01)

``SVTPrivacy``
    Implements the Sparse Vector Technique (SVT) for differential privacy.
    Uses Laplace noise and a threshold mechanism to selectively share weight updates
    while providing formal epsilon-differential privacy guarantees.

    Parameters: ``fraction`` (default 0.1), ``epsilon`` (default 0.1), ``noise_var`` (default 0.1)

``StatisticsPrivacyFilter``
    Applies privacy cleansing to federated statistics computations, ensuring that
    summary statistics shared across sites do not leak individual data points.

Usage Example
-------------

To add a privacy filter to a job, configure it as a ``task_result_filter`` on the client:

.. code-block:: python

    from nvflare.app_common.filters.percentile_privacy import PercentilePrivacy

    # In job configuration, add as a result filter:
    privacy_filter = PercentilePrivacy(percentile=10, gamma=0.01)

For filter configuration in job configs, see :ref:`Data Privacy & Filters <data_privacy_protection>`.

Combining DP with Other Privacy Mechanisms
==========================================

FLARE supports layered privacy:

- **DP + Homomorphic Encryption**: Apply DP filters before HE-encrypted aggregation for both
  input and output privacy
- **DP + Confidential Computing**: Run DP-protected training inside hardware TEEs for
  additional protection against infrastructure attacks
- **DP + Secure Aggregation**: Combine DP noise with secure aggregation protocols

See :doc:`/flare_security_overview` for an overview of all security mechanisms.

Resources
=========

- :doc:`Hello Differential Privacy </hello-world/hello-dp/index>` -- Complete DP-SGD example with Opacus
- :ref:`Data Privacy & Filters <data_privacy_protection>` -- Filter mechanism and configuration
- :ref:`Filters Programming Guide <filters>` -- How filters work in FLARE
- `Opacus Documentation <https://opacus.ai>`_ -- DP-SGD library for PyTorch
