# Replace with a reviewed digest-pinned base containing the provisioning
# NVFlare version, Python 3.11+, bash/coreutils and application dependencies.
FROM nvflare-client-base:REPLACE-WITH-REVIEWED-IMAGE-DIGEST

# Copy the reviewed module into this PRIVATE build context as documented.
# Ownership allows UID/GID 65532 to read a source file retained as mode 0600.
COPY --chown=65532:65532 tdx_acceptance.py /local/custom/tdx_acceptance.py
ENV PYTHONPATH=/local/custom

# CoCoPackager adds only this participant's signed kit and exact startup command.
