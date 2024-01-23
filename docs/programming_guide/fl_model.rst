.. _fl_model:

FLModel
=======

We define a standard data structure :mod:`FLModel<nvflare.app_common.abstract.FLModel>`
that captures the common attributes needed for exchanging learning results.

This is particularly useful when NVFlare system needs to exchange learning
information with external training scripts/systems.

The external training script/system only need to extract the required
information from received FLModel, run local training, and put the results
in a new FLModel to be sent back.

For a detailed explanation of each attributes, please refer to the API doc:
:mod:`FLModel<nvflare.app_common.abstract.FLModel>`
