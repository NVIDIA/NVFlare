# AWS EKS Auto Mode Test Tooling

This directory contains:

- `cluster.yaml` for a minimal EKS Auto Mode cluster
- `create_cluster.sh` and `delete_cluster.sh` for cluster lifecycle plus NVFlare storage bootstrap
- `inflate.yaml` for a tiny non-FLARE workload to verify the cluster can schedule pods

## Prerequisites

- `aws` CLI installed and authenticated
- `eksctl` installed
- `kubectl` installed

If you need to log in with AWS IAM Identity Center (SSO):

```bash
aws configure sso
aws sso login --profile my-sso-profile
export AWS_PROFILE=my-sso-profile
```

## Quick Start

Run these commands from the `devops/aws/eks` directory.

Edit `cluster.yaml` if you want a different cluster name or region.

Create the cluster:

```bash
./create_cluster.sh
```

The create script also:

- saves a dedicated kubeconfig to `.tmp/kubeconfigs/aws.yaml`
- creates the `auto-ebs-sc` default `StorageClass`
- creates an EFS filesystem plus mount targets for RWX storage
- installs the EFS CSI driver and creates the `efs-sc` `StorageClass`

Check the built-in Auto Mode node pools:

```bash
kubectl get nodepools
```

New Auto Mode clusters can start with zero nodes. Deploy the sample workload to force node provisioning:

```bash
kubectl apply -f inflate.yaml
kubectl get events -w
```

In another terminal, verify the pod and node:

```bash
kubectl get pods -o wide
kubectl get nodes
```

Delete the sample workload:

```bash
kubectl delete -f inflate.yaml
```

Delete the cluster when you are done:

```bash
./delete_cluster.sh
```

## NVFlare Deployment Setup

After `./create_cluster.sh`, the script has already done the cluster-side storage bootstrap:

- created `.tmp/kubeconfigs/aws.yaml`
- created `auto-ebs-sc` as the default RWO `StorageClass`
- created an EFS filesystem and mount targets
- installed the EFS CSI driver with the IAM wiring it needs
- created `efs-sc` for RWX workspace PVCs

That means you do not need to create the EBS or EFS `StorageClass` objects manually for the multicloud flow.

Notes:
- `auto-ebs-sc` uses the Auto Mode provisioner `ebs.csi.eks.amazonaws.com`
- `efs-sc` is the RWX class needed by `K8sJobLauncher`
- `delete_cluster.sh` also cleans up the EFS filesystem and its security group

Without RWX, use `ProcessJobLauncher` (in-process) or accept sequential `K8sJobLauncher` jobs.

### Container registry (ECR)

```bash
aws ecr create-repository --repository-name nvflare/nvflare --region <region>
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker push <account>.dkr.ecr.<region>.amazonaws.com/nvflare/nvflare:<tag>
```

### SELinux (Bottlerocket nodes)

EKS Auto Mode uses Bottlerocket with SELinux enforcing. All pods
sharing PVCs need `spc_t` security context:

```yaml
spec:
  securityContext:
    seLinuxOptions:
      type: spc_t
```

## Notes

- The sample workload is based on the AWS EKS Auto Mode `inflate` example.
- The `eks.amazonaws.com/compute-type: auto` selector makes this workload land on Auto Mode nodes.
- `cluster.yaml` remains the source of truth for the EKS cluster itself, while the scripts add the extra storage/bootstrap steps needed for NVFlare.
