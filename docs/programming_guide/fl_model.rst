.. _fl_model:

FLModel
=======

We define a standard data structure :mod:`FLModel<nvflare.app_common.abstract.fl_model>`
that captures the common attributes needed for exchanging learning results.

This is particularly useful when NVFlare system needs to exchange learning
information with external training scripts/systems.

The external training script or system only needs to extract the required
information from the received FLModel, run local training, and put the results
in a new FLModel to be sent back.

For a detailed explanation of each attribute, please refer to the API doc:
:mod:`FLModel<nvflare.app_common.abstract.fl_model>`
