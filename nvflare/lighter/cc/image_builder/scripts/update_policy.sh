#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 <Measurement>"
  exit 1
fi

POLICY_FILE=/shared/policy/policy.rego

cat <<EOF >> $POLICY_FILE

allow {
    input["submods"]["cpu"]["ear.veraison.annotated-evidence"]["snp"]["measurement"] == "$1"
}
EOF

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. $BASE_DIR/kbs_setup.sh

OUTPUT=$(sudo $KBS_CLIENT --url https://$TRUSTEE_HOST:$TRUSTEE_PORT --cert-file $ROOTCA config --auth-private-key $PRIVATE_KEY  set-resource-policy --policy-file $POLICY_FILE)

echo "$OUTPUT" | sed -n 's/.*policy: \(.*\).*/\1/p' | base64 -d
