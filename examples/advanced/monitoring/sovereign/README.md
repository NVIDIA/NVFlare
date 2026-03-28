# NVFlare Sovereign Quantum Profile

This package integrates key operational patterns from Sovereign-Mohawk-Proto into NVFlare monitoring:

- readiness gate automation for Prometheus/Grafana health
- quantum-proof path metrics for proof lifecycle and PQC posture
- provisioned Grafana dashboard for executive and SRE views

## What Is Included

1. `nvflare.metrics.quantum_proof_metrics_collector.QuantumProofMetricsCollector`
2. `nvflare_quantum_readiness_gate.py` automated PASS/FAIL readiness report
3. Grafana dashboard provisioning under `../setup/grafana/provisioning/dashboards`

## 1) Add Quantum Metrics Collector

Add the collector to `fed_server.json` and each `fed_client.json`:

```json
{
  "id": "quantum_proof_metrics_collector",
  "path": "nvflare.metrics.quantum_proof_metrics_collector.QuantumProofMetricsCollector",
  "args": {
    "tags": {
      "env": "prod",
      "region": "us-east"
    },
    "streaming_to_server": false,
    "expected_attestation_mode": "xmss",
    "expected_kex_mode": "x25519-mlkem768-hybrid",
    "pqc_migration_enabled": true,
    "pqc_lock_legacy_transfers": true
  }
}
```

If your clients stream metrics to the server, set `streaming_to_server=true` and keep `RemoteMetricsReceiver` on the server.

## 2) Start Monitoring Stack

From `examples/advanced/monitoring/setup`:

```bash
docker compose up -d
```

Endpoints:

- Grafana: `http://localhost:3000`
- Prometheus (basic-auth protected proxy): `http://localhost:9090`
- StatsD exporter metrics: `http://localhost:9102/metrics`

Security defaults in this setup:

- All published ports are bound to `127.0.0.1` only.
- Grafana admin password is read from `GRAFANA_ADMIN_PASSWORD` (defaults to `admin`).
- Prometheus is exposed through an NGINX basic-auth proxy.
  - Username: `nvflare`
  - Password: `nvflareprom`

The dashboard `NVFlare Sovereign Quantum Ops` is auto-provisioned.

## 3) Run Readiness Gate

From `examples/advanced/monitoring/sovereign`:

```bash
python3 nvflare_quantum_readiness_gate.py \
  --prom-url http://localhost:9090 \
  --prom-user nvflare \
  --prom-password nvflareprom \
  --grafana-url http://localhost:3000 \
  --output readiness-report.json
```

Exit code behavior:

- `0`: PASS
- `2`: FAIL

## Metric Contract

The collector emits these operational metrics:

- `quantum_path_ready`
- `quantum_pqc_controls_migration_enabled`
- `quantum_pqc_controls_legacy_lock_enabled`
- `quantum_proof_verify_count`
- `quantum_proof_verify_success_count`
- `quantum_proof_verify_failure_count`
- `quantum_proof_verify_time_taken`
- `quantum_proof_aggregation_count`
- `quantum_proof_aggregation_success_count`
- `quantum_proof_aggregation_time_taken`

This gives an auditable "path to quantum proofs" in operations dashboards and readiness pipelines.
