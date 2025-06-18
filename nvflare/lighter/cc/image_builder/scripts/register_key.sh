#!/usr/bin/env bash
if [ $# -ne 2 ]; then
  echo "Usage: $0 <vm_id> <key_file>"
  exit 1
fi

VM_ID=$1
ENC_KEY=$2

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. $BASE_DIR/kbs_setup.sh

URL_PATH=keys/root/${VM_ID}

OUTPUT=$($KBS_CLIENT --url https://$TRUSTEE_HOST:$TRUSTEE_PORT --cert-file $ROOTCA config --auth-private-key $PRIVATE_KEY set-resource --resource-file ${ENC_KEY} --path $URL_PATH)
echo $OUTPUT
KEY=$(echo $OUTPUT | sed -n 's/.*resource: \(.*\)/\1/p' | base64 --decode)
echo "Decoded key: ${KEY}"
