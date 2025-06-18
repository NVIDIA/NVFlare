#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 <vm_id>"
  exit 1
fi
VM_ID=$1

# Obtain measurement
REPORT=attestation-report.bin
./snpguest report ${REPORT} random-request-file.txt --random
MEASUREMENT=$(./snpguest display report ${REPORT} | sed ':a;N;$!ba;s/\n/ /g' | sed -n 's/.*Measurement:\(.*\)Host.*/\1/p' | sed 's/ //g' | xxd -r -p | base64)
echo "VM_ID: ${VM_ID} Measurement: ${MEASUREMENT}"

# Retrieve key
IP_ADDRESS=trustee-azsnptpm.eastus.cloudapp.azure.com
PORT=8999
URL_PATH=keys/root/${VM_ID}
ROOTCA=./rootCA.crt
KEY=$(./kbs-client --url https://$IP_ADDRESS:$PORT --cert-file $ROOTCA get-resource --path $URL_PATH | base64 --decode)
echo $KEY
KEY_HEAD="${KEY:0:4}"
KEY_TAIL="${KEY: -4}" 
MASKED_KEY="${KEY_HEAD} ... ${KEY_TAIL}"
echo "Retrieved key: ${MASKED_KEY}"

