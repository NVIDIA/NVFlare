#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CLUSTER_NAME="$(grep 'name:' "${SCRIPT_DIR}/cluster.yaml" | head -1 | awk '{print $2}')"
REGION="$(grep 'region:' "${SCRIPT_DIR}/cluster.yaml" | awk '{print $2}')"

# Clean up EFS resources before deleting the cluster
EFS_IDS="$(aws efs describe-file-systems --region "${REGION}" \
  --query "FileSystems[?Tags[?Key=='Name'&&Value=='${CLUSTER_NAME}-nvflare']].FileSystemId" \
  --output text 2>/dev/null || true)"

for EFS_ID in ${EFS_IDS}; do
  echo "Cleaning up EFS ${EFS_ID} ..."
  # Delete mount targets first
  MT_IDS="$(aws efs describe-mount-targets --file-system-id "${EFS_ID}" --region "${REGION}" \
    --query 'MountTargets[].MountTargetId' --output text 2>/dev/null || true)"
  for MT in ${MT_IDS}; do
    aws efs delete-mount-target --mount-target-id "${MT}" --region "${REGION}" 2>/dev/null || true
  done
  # Poll until all mount targets are gone (async; can take well over 15s)
  for i in $(seq 1 30); do
    MT_COUNT="$(aws efs describe-mount-targets --file-system-id "${EFS_ID}" --region "${REGION}" \
      --query 'length(MountTargets)' --output text 2>/dev/null || echo 1)"
    [[ "${MT_COUNT}" == "0" ]] && break
    sleep 5
  done
  if ! aws efs delete-file-system --file-system-id "${EFS_ID}" --region "${REGION}"; then
    echo "ERROR: Failed to delete EFS ${EFS_ID} (mount targets still present?)" >&2
    exit 1
  fi
  echo "Deleted EFS ${EFS_ID}"
done

# Delete the EFS security group
SG_ID="$(aws ec2 describe-security-groups --region "${REGION}" \
  --filters "Name=group-name,Values=${CLUSTER_NAME}-efs-sg" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)"
if [[ -n "${SG_ID}" && "${SG_ID}" != "None" ]]; then
  aws ec2 delete-security-group --group-id "${SG_ID}" --region "${REGION}" 2>/dev/null || true
  echo "Deleted security group ${SG_ID}"
fi

eksctl delete cluster -f "${SCRIPT_DIR}/cluster.yaml" --wait
