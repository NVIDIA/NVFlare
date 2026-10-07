# Maintenance boundary

The host/account-specific lab teardown scripts are deliberately not distributed
in this public package. They encode destructive paths and assumptions about
ownership that do not generalize safely. There is no packaged `90-*` purge or
cluster teardown command; do not run a command copied from an old lab record.
The original operational scripts remain outside this public example.

For a disposable deployment, the simplest clean reset is to recreate the
machine using its infrastructure owner's normal process, after explicitly
approving destruction and handling required backups. Deleting a Pod alone
does not remove container images, keys, services, Kubernetes or containerd.

For selective cleanup, the machine owner must first inventory and approve exact
resources and dependencies: workload images and shared Docker build cache on
admin; registry storage, TLS keys, KBS storage, authorization, RVPS references
and backups on secure services; Kubernetes workloads, Kata/GPU Operator,
containerd, mounts and networking on cluster hosts. Preserve anything belonging
to another workload or user. Do not use blanket recursive directory deletion.

The rehearsal scripts clean up their own temporary Pod namespace and registry
container; they retain private evidence and build artifacts for review. Protect
or deliberately dispose of those outputs under the organization's retention
policy. Revoke publisher credentials and retire workload authorizations when
no longer needed. A later fresh installation uses newly generated certificates
and new workload authorization, not a published snapshot of private state.
