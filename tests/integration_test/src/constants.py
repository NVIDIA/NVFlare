# need to be consistent with provision
RESOURCE_CONFIG = "resources.json"
DEFAULT_RESOURCE_CONFIG = "resources.json.default"
SERVER_NVF_CONFIG = "fed_server.json"
CLIENT_NVF_CONFIG = "fed_client.json"


FILE_STORAGE = "nvflare.app_common.storages.filesystem_storage.FilesystemStorage"

SERVER_SCRIPT = "nvflare.private.fed.app.server.server_train"
CLIENT_SCRIPT = "nvflare.private.fed.app.client.client_train"


# provision
PROVISION_SCRIPT = "nvflare.cli provision"

# preflight check
PREFLIGHT_CHECK_SCRIPT = "nvflare.cli preflight_check"
