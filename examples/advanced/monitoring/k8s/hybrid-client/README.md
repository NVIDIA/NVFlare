# Hybrid Setup 2: External Server, Kubernetes Client

This directory contains a reference client manifest for the common hybrid deployment pattern:

- NVFLARE server runs outside Kubernetes
- one or more NVFLARE clients run inside Kubernetes
- monitoring uses setup 2, so client metrics stream to the server

## Why Setup 2 Is Recommended

In this topology, the Kubernetes client does not need direct access to `statsd-exporter`, Prometheus, or Grafana.

Instead:

1. the client collects metrics locally
2. the client streams metrics to the NVFLARE server through the existing FL connection
3. the server forwards those metrics to the monitoring stack

This keeps the Kubernetes client simple and avoids exposing the StatsD port to every client environment.

## What the Client Needs

The Kubernetes client only needs:

- a startup kit mounted into the pod
- network reachability to the external NVFLARE server
- setup-2 client resources that stream metrics to the server

The tracked setup-2 client example is:

- [../../jobs/setup-2/local_config/site-1/resources.json](../../jobs/setup-2/local_config/site-1/resources.json)

That configuration uses `SysMetricsCollector` with `streaming_to_server: true` plus `ConvertToFedEvent`.

## What the Server Needs

The external NVFLARE server needs:

- setup-2 server resources with `RemoteMetricsReceiver` and `StatsDReporter`
- reachability to the monitoring stack chosen for your deployment

The tracked setup-2 server example is:

- [../../jobs/setup-2/local_config/server/resources.json](../../jobs/setup-2/local_config/server/resources.json)

Only the server needs direct `StatsDReporter` connectivity to the monitoring stack in this pattern.

## Hostname Routing

Keep the signed startup kit unchanged when possible.

If the signed `fed_client.json` already references a hostname that resolves correctly from the cluster, the pod can connect without extra routing changes.

If the expected hostname does not resolve inside the cluster, preserve the signed kit and add hostname routing outside the kit. The example manifest uses `hostAliases` for that purpose.

## Example Manifest

- [10-client-pod-hostaliases.yaml](10-client-pod-hostaliases.yaml)

Replace these placeholders before applying it:

- `<client-namespace>`
- `<client-site-name>`
- `<nvflare-runtime-image>`
- `<external-server-ip>`
- `<server-hostname-from-signed-kit>`
- `<path-to-provisioned-client-kit>`

## Practical Consequences

With setup 2 in this hybrid topology:

- multiple Kubernetes clients can all stream metrics through the same server
- one central monitoring stack is enough
- clients do not need per-client monitoring stacks unless you intentionally choose setup 3
- server metrics stay at the server side; metrics are aggregated in the supported direction of client to server to monitoring

This reference manifest follows the same signed-kit-safe client pod pattern validated in the in-cluster Kubernetes flow. The exact external server hostname, IP, and firewall path still need to be verified in your environment.
