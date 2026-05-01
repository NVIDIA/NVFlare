# AWS EKS Auto Mode

Cluster lifecycle + NVFlare storage bootstrap.

## Prereqs

`aws` (authenticated), `eksctl`, `kubectl`. SSO: `aws configure sso && aws sso login --profile <p>`.

## Create

Edit `cluster.yaml` for a different name/region, then:

```bash
./create_cluster.sh
```

Script:
- saves kubeconfig to `.tmp/kubeconfigs/aws.yaml`
- creates EFS filesystem + mount targets + `efs-sc` StorageClass (RWX)
- installs the EFS CSI driver with its IAM role
- uses built-in `auto-ebs-sc` for RWO

## Verify + smoke test

```bash
kubectl get nodepools
kubectl apply -f inflate.yaml && kubectl get pods -w   # triggers node provisioning
kubectl delete -f inflate.yaml
```

## Delete

```bash
./delete_cluster.sh
```

Also cleans up the EFS filesystem and its security group.

## Notes

- Bottlerocket nodes run SELinux enforcing; pods sharing PVCs need
  `securityContext.seLinuxOptions.type: spc_t` (deploy tool does this
  automatically via the YAML config).
- Push images to ECR:
  ```bash
  aws ecr create-repository --repository-name nvflare/nvflare --region <region>
  aws ecr get-login-password --region <region> | \
    docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
  ```
