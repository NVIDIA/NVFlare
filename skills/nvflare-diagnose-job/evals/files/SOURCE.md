# Diagnosis Eval Fixtures

These fixtures are minimal synthetic log snippets derived from common NVFLARE
failure shapes. They are not copied from customer jobs and contain no private
site data.

`transfer_progress_timeout.log` is synthetic evidence for a large-transfer
timeout warning where later streaming progress indicates a retryable congestion
case rather than a dead peer.
