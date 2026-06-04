# NVFlare DevOps Examples

This directory contains example scripts for quickly testing NVFlare deployment
flows on Kubernetes, OpenShift, and managed cloud clusters. They are intended
for local development, smoke testing, demos, and learning.

These scripts are not production quality. They are not a hardened deployment
blueprint and do not replace site-specific review for security, networking,
identity, storage, monitoring, backups, upgrade strategy, cost controls, or
operations.

## Scope

Use these examples to create or target temporary test clusters, build and push
a test NVFlare image, deploy a small NVFlare system, inspect it, and tear it
down. They assume you already have an NVFlare development environment and the
required Kubernetes, OpenShift, or cloud CLIs configured for the target
clusters, accounts, or projects.

Before running a deployment, copy or edit `examples/devops/multicloud/all-clouds.yaml`
and replace the placeholder image tag, kubeconfig inputs, namespaces, storage
classes, and participants for the clusters you want to test.

## Layout

- `multicloud/` - YAML-driven NVFlare deployment, status, dashboard, and image
  build/push helpers.
- `openshift/` - OpenShift-specific deployment guide and helper scripts using
  the Kubernetes runtime support.
- `gcp/gke/`, `aws/eks/`, `azure/aks/` - cloud cluster setup scripts and notes.
- `examples/devops/.tmp/` - local generated kubeconfigs and state; not intended for
  commit.

## Typical Flow

Run these commands from the NVFlare repository root after updating the config:

```bash
python examples/devops/multicloud/fetch_kubeconfigs.py --config examples/devops/multicloud/all-clouds.yaml
python examples/devops/multicloud/build_and_push.py --config examples/devops/multicloud/all-clouds.yaml
python examples/devops/multicloud/deploy.py --config examples/devops/multicloud/all-clouds.yaml up
python examples/devops/multicloud/k8sview.py --config examples/devops/multicloud/all-clouds.yaml
```

To tear down a deployed test system:

```bash
python examples/devops/multicloud/deploy.py --config examples/devops/multicloud/all-clouds.yaml down
```
