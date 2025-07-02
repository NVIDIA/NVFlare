# Trustee and KBS client configuration

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../kbs" && pwd)"

TRUSTEE_HOST=trustee-azsnptpm.eastus.cloudapp.azure.com
TRUSTEE_PORT=8999
PRIVATE_KEY="$BASE_DIR/private.key"
ROOTCA="$BASE_DIR/rootCA.crt"
KBS_CLIENT="$BASE_DIR/kbs-client"
