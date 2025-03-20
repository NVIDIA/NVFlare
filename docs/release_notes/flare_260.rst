**************************
What's New in FLARE v2.6.0
**************************


**********************************
Migration to 2.6.0: Notes and Tips
**********************************


For PTClientAPILauncherExecutor and PTInProcessClientAPIExecutor
FLARE 2.6.0 introduces significant changes to the "params_exchange_format" argument in PTClientAPILauncherExecutor and PTInProcessClientAPIExecutor. These changes impact how data is exchanged between the client script and NVFlare.

### Changes in params_exchange_format
In previous versions, setting "params_exchange_format" to "pytorch" indicated that the client was using a PyTorch tensor on the third-party side. In this case, the tensor would be converted to a NumPy array before being sent back to NVFlare.

With the improvements introduced in FLARE 2.6.0, which now natively support PyTorch tensors during transmission, the meaning of "params_exchange_format" = "pytorch" has changed. Now, this setting directly sends PyTorch tensors to NVFlare without converting them to NumPy arrays.

### Action Required
To maintain the previous behavior (where PyTorch tensors are converted to NumPy arrays), you will need to explicitly set "params_exchange_format" to "numpy".

