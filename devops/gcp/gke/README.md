# Google Cloud GKE Autopilot

Cluster lifecycle + NVFlare storage bootstrap.

## Prereqs

`gcloud` (authenticated, project selected), `kubectl`, `gke-gcloud-auth-plugin`.

```bash
gcloud auth login
gcloud config set project <project>
gcloud services enable container.googleapis.com
gcloud components install gke-gcloud-auth-plugin
```

## Create

```bash
./create_cluster.sh
```

Defaults: `CLUSTER_NAME=gke-auto-test`, `LOCATION=us-central1`, `NETWORK_NAME=gke-test`. Override with env vars:

```bash
PROJECT_ID=<p> CLUSTER_NAME=<c> LOCATION=<r> ./create_cluster.sh
```

Script:
- saves kubeconfig to `.tmp/kubeconfigs/gcp.yaml`
- enables `file.googleapis.com`
- creates a `filestore-rwx` StorageClass bound to the cluster VPC

## Verify + smoke test

```bash
kubectl get nodes
kubectl apply -f inflate.yaml && kubectl get pods -w
kubectl delete -f inflate.yaml
```

## Delete

```bash
./delete_cluster.sh
```

Removes the VPC only if this script created it.

## Notes

- Filestore: 1 TiB minimum (~$200/mo), 3–5 min to provision. If you
  hit zone capacity errors, pin a zone:
  `FILESTORE_ZONE=us-central1-a ./create_cluster.sh`.
- Push images to Artifact Registry:
  ```bash
  gcloud artifacts repositories create nvflare \
    --repository-format=docker --location=us-central1 --project=<p>
  gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
  # grant node SA pull access:
  PROJECT_NUM=$(gcloud projects describe <p> --format="value(projectNumber)")
  gcloud artifacts repositories add-iam-policy-binding nvflare \
    --project=<p> --location=us-central1 \
    --member="serviceAccount:${PROJECT_NUM}-compute@developer.gserviceaccount.com" \
    --role="roles/artifactregistry.reader"
  ```
