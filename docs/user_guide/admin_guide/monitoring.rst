.. _monitoring:

Monitoring
**********

NVIDIA FLARE provides system monitoring capabilities to track job and system lifecycle metrics for federated learning jobs. This feature enables comprehensive system-level monitoring and visualization of FLARE operations.

Overview
========

The monitoring system focuses on system-level metrics rather than training metrics, providing insights into:

* Job lifecycle events
* System performance metrics
* Resource utilization
* Network statistics

This differs from machine learning experiment tracking by focusing on system-level metrics rather than training metrics.

Key Components
--------------

The monitoring system leverages the following components:

* StatsD Exporter: Collects and exports system metrics
* Prometheus: Scrapes and stores the metrics
* Grafana: Visualizes the collected metrics

These components work together to provide a complete monitoring solution:

1. FLARE job and system events are captured
2. StatsD Exporter processes these events
3. Prometheus scrapes the metrics from StatsD Exporter
4. Grafana provides visualization and dashboards

Configuration
-------------

For detailed configuration instructions and setup steps, please refer to the :github_nvflare_link:`Monitoring README <examples/advanced/monitoring/README.md>`.

The configuration involves:

* Setting up StatsD Exporter
* Configuring Prometheus
* Installing and configuring Grafana
* Setting up FLARE monitoring components

Visualization
-------------

Grafana dashboards provide visual representations of the collected metrics, including:

* Job status and progress
* System resource usage
* Network statistics
* Component health

For detailed information about available dashboards and metrics, please refer to the :github_nvflare_link:`Monitoring README <examples/advanced/monitoring/README.md>`.

Summary
-------

The monitoring system provides:

* Real-time visibility into FLARE operations
* System-level metrics tracking
* Comprehensive visualization capabilities
* Integration with industry-standard monitoring tools

For more detailed information about setup, configuration, and usage, please refer to the :github_nvflare_link:`Monitoring README <examples/advanced/monitoring/README.md>`. 