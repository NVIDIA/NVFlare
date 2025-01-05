# Hello PyTorch
Please see [hello-pytorch](../../../../hello-world/hello-pt/README.md) for details related how to run hello-pt

## Prepare Code

Create a bash script to copy the hello-world example to the current directory:

```bash
#!/bin/bash

# Copy hello-world example to the current directory
cp -r ../../../../hello-world/hello-pt/* .
```

Save this script as `prepare_code.sh` and run it to set up the example.

## Modify `fedavg_script_runner_pt.py`

We will modify the `fedavg_script_runner_pt.py` using the Job API for the two setup scenarios described in the `../README.md` file. We will ignore the separate monitoring system setup case (setup 3) as the process is very similar, just needs more than one monitoring system. and each monitoring system can only observe its own metrics information. 

- **Setup 1**: All sites share the same `statsd-reporter` with the same host and port.
- **Setup 2**: Only the server site has the `statsd-reporter`, clients forward metrics to the server.


## Example Job Configuration


Here is an example of how to add components for `JobMetricsCollector` and `statsd-reporter` using the Job API: Assuming all sites share the same monitoring system (setup 2)


The `JobMetricsCollector` and `statsd-reporter` will be part of the FLARE job configuration. The component configuration can be added in the job workflow and can be specified directly in the JSON configuration or using the Job API as shown in the hello-pt example. Depending on the setup, the `statsd-reporter` will have different hosts and ports (setup 1) or the same host and port (setup 2). In setup 3, the `statsd-exporter` will only be needed on the server side, not on the client site.

### Setup 1: All Sites Share the Same Monitoring System

In this setup, all sites (server and clients) will share the same monitoring system with the same host and port.

```python
job = FedAvgJob(...)

# Shared StatsD Reporter
shared_metrics_reporter = StatsDReporter(host="localhost", port=9125)

# Server configuration
server_metrics_collector = JobMetricsCollector(tags={"site": "server", "env": "dev"})
job.to_server(server_metrics_collector, id="server_job_metrics_collector")
job.to_server(shared_metrics_reporter, id="shared_statsd_reporter")

# Client configuration
for i in range(n_clients):
    client_site = f"site-{i + 1}"
    executor = ScriptRunner(script=train_script, script_args="")
    job.to(executor, client_site)

    client_metrics_collector = JobMetricsCollector(tags={"site": client_site, "env": "dev"})
    job.to(client_metrics_collector, target=client_site, id=f"{client_site}_job_metrics_collector")
    job.to(shared_metrics_reporter, target=client_site, id=f"{client_site}_shared_statsd_reporter")

job.export_job("/tmp/nvflare/jobs/job_config")
job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
```

### Setup 2: Only Server Site Has the Monitoring System

In this setup, only the server site has the monitoring and clients forward metrics to the server.

```python
job = FedAvgJob(...)

# Server configuration
server_metrics_reporter = StatsDReporter(host="localhost", port=9125)
server_metrics_collector = JobMetricsCollector(tags={"site": "server", "env": "dev"})
job.to_server(server_metrics_collector, id="server_job_metrics_collector")
job.to_server(server_metrics_reporter, id="server_statsd_reporter")

# Client configuration
for i in range(n_clients):
    client_site = f"site-{i + 1}"
    executor = ScriptRunner(script=train_script, script_args="")
    job.to(executor, client_site)

    client_metrics_collector = JobMetricsCollector(tags={"site": client_site, "env": "dev"})
    job.to(client_metrics_collector, target=client_site, id=f"{client_site}_job_metrics_collector")

job.export_job("/tmp/nvflare/jobs/job_config")
job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
```

 
This example demonstrates how to configure the `JobMetricsCollector` and `statsd-reporter` for both the server and client sites within the job workflow.

Make sure to replace the placeholder logic with the actual implementation as per your requirements.

This example focused on how to setup additional plugin components to enable statsd with prometheus + grafana monitoring


We have make a docker-compose file to use docker to simulate different services: stats-exporter, prometheus and grafana. 


## start up monitoring system

Assuming you already have Docker and Docker Compose installed, you can use the provided `docker-compose.yml` file to set up StatsD Exporter, Prometheus, and Grafana.

### Steps:

1. Navigate to the setup directory:
    ```bash
    cd setup
    ```

2. Start the services using Docker Compose:
    ```bash
    docker-compose up -d
    ```

3. To stop the services, run:
    ```bash
    docker-compose down
    ```

**Note:** The StatsD Exporter port is using 9125 (not 8125).

## Start Up POC

1. Prepare POC:
    ```bash
    nvflare poc prepare
    ```
this will prepare 1 server and 2 clients ("site-1", "site-2") and one admin console client(admin@nvidia.com)
you can examine the output directory: ```/tmp/nvflare/poc/example_project/prod_00```


2. Start POC:
    ```bash
    nvflare poc start -ex admin@nvidia.com
    ```
    This will exclude the admin console service.

3. Run Job 
    see run job section
    
4. Stop POC:
    After you complete the job run, you can stop the POC by:

    ```bash
    nvflare poc stop
    ```

## run job via CLI 

To run the job from the command line, use the following command:

```bash
nvflare job submit -j <job folder>
```
 


