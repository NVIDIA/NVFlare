# NVFLARE Prometheus Monitoring Setup

This guide describes how to set up NVFLARE using the `JobMetricsCollector` and `SysMetricsCollector` plugins to enable metrics publishing to StatsD Exporter, Prometheus, and Grafana.

## Setup Types

### 1. Individual Monitoring Systems for Each Site

Each client site and the server site will have its own monitoring system, including StatsD Exporter, Prometheus, and Grafana.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on each site.
2. **Configure StatsD Exporter** to receive metrics from NVFLARE.
3. **Configure Prometheus** to scrape metrics from StatsD Exporter.
4. **Set up Grafana** to visualize the metrics from Prometheus.

### 2. Shared Monitoring System for All Sites

All sites will share the same monitoring system.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on a central monitoring server.
2. **Configure StatsD Exporter** to receive metrics from all NVFLARE sites.
3. **Configure Prometheus** to scrape metrics from the central StatsD Exporter.
4. **Set up Grafana** to visualize the metrics from Prometheus.

### 3. Clients Forward Metrics to Server Site

Clients will forward (stream) the metrics to the server site, and the server site will publish to the monitoring system, which will have a consolidated view of all metrics.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on the server site.
2. **Configure clients** to forward metrics to the server site.
3. **Configure StatsD Exporter** on the server site to receive metrics from all clients.
4. **Configure Prometheus** on the server site to scrape metrics from StatsD Exporter.
5. **Set up Grafana** on the server site to visualize the metrics from Prometheus.

## Plugin Configuration

### JobMetricsCollector and StatsD Reporter

- **Setup 1**: All sites share the same `statsd-reporter` with the same host and port.
- **Setup 2**: Only the server site has the `statsd-reporter`, clients forward metrics to the server.
- **Setup 3**: Each site has its own `statsd-reporter` with different hosts and ports.

Add the following configuration to your NVFLARE job configuration file:

```json
{
    "id": "client_1_job_metrics_collector",
    "path": "nvflare.metrics.job_metrics_collector.JobMetricsCollector",
    "args": {
        "tags": {
            "site": "site_1",
            "env": "dev"
        }
    }
},
{
    "id": "client_1_statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
}
```

### SysMetricsCollector

The `SysMetricsCollector` is for the client and server parent process and must be configured in the local resources configuration file for each site. This cannot be specified from the Job API.

Add the following configuration to your NVFLARE system configuration file:
```json
{
    "id": "sys_metrics_collector",
    "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
}


Replace `<statsd_exporter_host>` and `<statsd_exporter_port>` with the appropriate values for your setup.

## Conclusion

By following the steps outlined above, you can set up NVFLARE to publish metrics to StatsD Exporter, Prometheus, and Grafana, using one of the three setup types that best fits your needs.
