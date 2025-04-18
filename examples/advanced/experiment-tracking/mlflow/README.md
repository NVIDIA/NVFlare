# MLflow Experiment Tracking with NVIDIA FLARE
 
## Example 1: Centralized MLflow Tracking

In this example:

- The client uses the MLflow writer to collect training metrics.
- Metrics are streamed to the server-side receiver.
- The server receiver then sends the metrics to an MLflow server (either remote or local).
- The same metrics can optionally be concurrently sent to TensorBoard for visualization.

**Use case**: When a consolidated, centralized view of all clients' training progress is required.

---

## Example 2: Site-Specific MLflow Tracking

In this setup:

- The client uses the MLflow writer to collect training metrics.
- Metrics are sent to a client-side receiver instead of a centralized server.
- These metrics are then delivered to a client site hosted MLflow server or written to local storage.

**Use case**: When each client site wants to track and manage its own training metrics independently.

---

## Configuration Flexibility

FLARE allows seamless switching between centralized and decentralized experiment tracking by modifying only the configuration:

- The training code remains unchanged.
- You can control:
  - Where the metrics are sent (server or client site).
  - Which experiment tracking framework is used.

This flexible design enables easy integration with different observability platforms, tailored to your deployment needs.
