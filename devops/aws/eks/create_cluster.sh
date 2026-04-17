#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Extract cluster name and region from cluster.yaml
CLUSTER_NAME="$(grep 'name:' "${SCRIPT_DIR}/cluster.yaml" | head -1 | awk '{print $2}')"
REGION="$(grep 'region:' "${SCRIPT_DIR}/cluster.yaml" | awk '{print $2}')"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

eksctl create cluster -f "${SCRIPT_DIR}/cluster.yaml"

# Save kubeconfig for multicloud deploy scripts
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${REPO_ROOT}/.tmp/kubeconfigs"
aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}" \
  --kubeconfig "${REPO_ROOT}/.tmp/kubeconfigs/aws.yaml"
echo "Kubeconfig saved to .tmp/kubeconfigs/aws.yaml"

# ---------------------------------------------------------------------------
# EBS StorageClass (RWO) — EKS Auto Mode has no usable default
# ---------------------------------------------------------------------------
kubectl apply -f - <<'EOF'
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: auto-ebs-sc
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: ebs.csi.eks.amazonaws.com
volumeBindingMode: WaitForFirstConsumer
parameters:
  type: gp3
  encrypted: "true"
EOF

# ---------------------------------------------------------------------------
# EFS (RWX) — needed for K8sJobLauncher workspace PVC sharing
# ---------------------------------------------------------------------------

# Get the VPC and subnets used by the cluster
VPC_ID="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query 'cluster.resourcesVpcConfig.vpcId' --output text)"
SUBNET_IDS="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query 'cluster.resourcesVpcConfig.subnetIds' --output text)"

# Create EFS filesystem
EFS_ID="$(aws efs create-file-system \
  --region "${REGION}" \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --encrypted \
  --tags Key=Name,Value="${CLUSTER_NAME}-nvflare" \
  --query 'FileSystemId' --output text)"
echo "Created EFS filesystem: ${EFS_ID}"

# Wait for EFS to be available
echo "Waiting for EFS to be available ..."
aws efs describe-file-systems --file-system-id "${EFS_ID}" --region "${REGION}" \
  --query 'FileSystems[0].LifeCycleState' --output text
for i in $(seq 1 30); do
  STATE="$(aws efs describe-file-systems --file-system-id "${EFS_ID}" --region "${REGION}" \
    --query 'FileSystems[0].LifeCycleState' --output text)"
  if [[ "${STATE}" == "available" ]]; then break; fi
  sleep 5
done

# Create a security group allowing NFS from the VPC CIDR
VPC_CIDR="$(aws ec2 describe-vpcs --vpc-ids "${VPC_ID}" --region "${REGION}" \
  --query 'Vpcs[0].CidrBlock' --output text)"
SG_ID="$(aws ec2 create-security-group \
  --group-name "${CLUSTER_NAME}-efs-sg" \
  --description "NFS access for NVFlare EFS" \
  --vpc-id "${VPC_ID}" \
  --region "${REGION}" \
  --query 'GroupId' --output text)"
aws ec2 authorize-security-group-ingress \
  --group-id "${SG_ID}" \
  --protocol tcp --port 2049 \
  --cidr "${VPC_CIDR}" \
  --region "${REGION}"

# Create mount targets in each subnet
for SUBNET in ${SUBNET_IDS}; do
  aws efs create-mount-target \
    --file-system-id "${EFS_ID}" \
    --subnet-id "${SUBNET}" \
    --security-groups "${SG_ID}" \
    --region "${REGION}" 2>/dev/null || true
done
echo "EFS mount targets created"

# Create IAM role for EFS CSI driver (Pod Identity / IRSA)
OIDC_ID="$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${REGION}" \
  --query 'cluster.identity.oidc.issuer' --output text | sed 's|https://||')"

# Ensure OIDC provider exists
aws iam list-open-id-connect-providers --query "OpenIDConnectProviderList[?ends_with(Arn, '${OIDC_ID}')].Arn" --output text | grep -q . || \
  eksctl utils associate-iam-oidc-provider --cluster "${CLUSTER_NAME}" --region "${REGION}" --approve

# Create the EFS CSI driver service account with IAM role
eksctl create iamserviceaccount \
  --cluster "${CLUSTER_NAME}" \
  --region "${REGION}" \
  --namespace kube-system \
  --name efs-csi-controller-sa \
  --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy \
  --approve \
  --override-existing-serviceaccounts 2>/dev/null || true

# Install EFS CSI driver add-on
echo "Installing EFS CSI driver add-on ..."
aws eks create-addon \
  --cluster-name "${CLUSTER_NAME}" \
  --addon-name aws-efs-csi-driver \
  --region "${REGION}" || { echo "ERROR: EFS CSI driver install failed" >&2; exit 1; }

echo "Waiting for EFS CSI driver to become active ..."
if ! aws eks wait addon-active \
  --cluster-name "${CLUSTER_NAME}" \
  --addon-name aws-efs-csi-driver \
  --region "${REGION}"; then
  echo "ERROR: EFS CSI driver did not become active" >&2
  exit 1
fi

# Create EFS StorageClass
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: ${EFS_ID}
  directoryPerms: "700"
EOF

echo "EFS setup complete: ${EFS_ID}"
echo "Use storageClassName: efs-sc for RWX PVCs"
