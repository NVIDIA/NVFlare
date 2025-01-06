# NVFLARE Prometheus Monitoring Setup

This guide describes how to set up NVFLARE using the `JobMetricsCollector` and `SysMetricsCollector` plugins to enable metrics publishing to StatsD Exporter, Prometheus, and Grafana.

## Setup Types

### 1. Shared Monitoring System for All Sites

All sites will share the same monitoring system.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on a central monitoring server.
2. **Configure StatsD Exporter** to receive metrics from all NVFLARE sites.
3. **Configure Prometheus** to scrape metrics from the central StatsD Exporter.
4. **Set up Grafana** to visualize the metrics from Prometheus.

### 2. Clients Forward Metrics to Server Site

Clients will forward (stream) the metrics to the server site, and the server site will publish to the monitoring system, which will have a consolidated view of all metrics.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on the server site.
2. **Configure clients** to forward metrics to the server site.
3. **Configure StatsD Exporter** on the server site to receive metrics from all clients.
4. **Configure Prometheus** on the server site to scrape metrics from StatsD Exporter.
5. **Set up Grafana** on the server site to visualize the metrics from Prometheus.


### 3. Individual Monitoring Systems for Each Site

Each client site and the server site will have its own monitoring system, including StatsD Exporter, Prometheus, and Grafana.

#### Steps:
1. **Install StatsD Exporter, Prometheus, and Grafana** on each site.
2. **Configure StatsD Exporter** to receive metrics from NVFLARE.
3. **Configure Prometheus** to scrape metrics from StatsD Exporter.
4. **Set up Grafana** to visualize the metrics from Prometheus.


## NVFLARE Monitoring Components Configuration

### Components Overview

We have several components to use depending on the type of metrics as well as the setups:

1. **StatsDReporter**: This component will post the collected metrics to StatsD Exporter service.
2. **JobMetricsCollector**: This component collects job-level metrics and publishes them to the databus. It can be added to the workflow components on both client and server sites.
3. **SysMetricsCollector**: This component collects system-level metrics running in the parent process of the server and clients. The metrics will be published to the databus.
4. **RemoteMetricsReceiver**: This component receives the federated metrics streamed from client sides and publish the metriics. 
5. **ConvertToFedEvent**: This component converts local events to federated events from client to server.


### Components Configuration

We will describe the component configuration in the following sections, but note some of the Job level configurations can be auto-generated from Job API, which will be described in the [job example](jobs/README.md).

>> sidebar note:

> The NVIDIA FLARE json component configration is very simple, it consists the following patterns 
```{ 
   "id": "<any string to represent compoent>"
   "path": "<fully qualified classpath>",
    "args": {
        <constructor arguments key, value pairs>
    }
```

#### 1. Shared Monitoring System for All Sites

In this setup, all sites post the metrics to the common StatsD Exporter service. Therefore, all sites will need StatsD Exporter with the same host and port. Additionally, all sites will need both JobMetricsCollector and SysMetricsCollector components. 

We don't need streaming metrics, so the ConvertToFedEvent and RemoteMetricsReceiver components are not needed.

To add Job Metrics Collector, we will add component in job configurations ```fed_config_client.json``` and ```fed_config_server.json```. For example


```fed_config_client.json```

```json
{
    "id": "job_metrics_collector",
    "path": "nvflare.metrics.job_metrics_collector.JobMetricsCollector",
    "args": {
        "tags": {
            "site": "site_1",
            "env": "dev"
        }
    }
},
{
    "id": "statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
}
``` 


```fed_config_server.json```

```json
{
    "id": "job_metrics_collector",
    "path": "nvflare.metrics.job_metrics_collector.JobMetricsCollector",
    "args": {
        "tags": {
            "site": "server",
            "env": "dev"
        }
    }
},
{
    "id": "statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
}
``` 

tags can be key, value pair, they are used for group metrics in the report. Here we used "site" to indicate origin of the metrics, the "dev" env. to indicating the dev environment. 



The `SysMetricsCollector` is for the client and server parent process and must be configured in the local resources configuration file for each site. This cannot be specified from the Job API.

In the ```<startup>/<site-name>/local/resources.json.default```

we can create a customized local configuration

```<startup>/<site-name>/local/resources.json```

by rename ```resources.json.default``` to ```resources.json```

in ```<startup>/<site-name>/local/resources.json```

Add the following configuration:

```json
{
    "id": "sys_metrics_collector",
    "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
    "args": {
        "tags": {
            "site": "<site>",
            "env": "dev"
        }
    }
}, 
{
    "id": "statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
}
```

Replace `<statsd_exporter_host>` and `<statsd_exporter_port>` with the appropriate values for your setup.



#### 2. Clients Forward Metrics to Server Site
In this setup, all client-side metrics will not directly post to the StatsD Exporter. Instead, the metrics are streamed to the server site. Therefore, the client side will need the following components:

- **JobMetricsCollector**
- **SysMetricsCollector**
- **ConvertToFedEvent**

On the server side, we will need:

- **StatsDReporter**
- **JobMetricsCollector**
- **SysMetricsCollector**
- **RemoteMetricsReceiver**

In ```fed_config_client.json```,

```json
{
    "id": "job_metrics_collector",
    "path": "nvflare.metrics.job_metrics_collector.JobMetricsCollector",
    "args": {
        "tags": {
            "site": "site_1",
            "env": "dev"
        }, 
        "streaming_to_server": True
    }
},
{
    "id": "event_convertor",
    "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
    "args": {
      "events_to_convert": ["metrics_event"]
    }
}
```


```fed_config_server.json```

```json
{
    "id": "sys_metrics_collector",
    "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
    "args": {
        "tags": {
            "site": "server",
            "env": "dev"
        }
    }
}, 
{
    "id": "statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
},
{
    "id": "remote_metrics_receiver",
    "path": "nvflare.metrics.remote_metrics_reciever.RemoteMetricsReceiver",
     "args": {
         "events": ["fed.metrics_event"]
     }
} 
```


in client side (site-1, site-2) ```<startup>/<site-name>/local/resources.json```


```json
{
    "id": "sys_metrics_collector",
    "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
    "args": {
        "tags": {
            "site": "<site>",
            "env": "dev"
        },
        
        "streaming_to_server": True

    }
}, 
{
    "id": "event_convertor",
    "path": "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
    "args": {
      "events_to_convert": ["metrics_event"]
    }
}
 
```

in server side (such "server") ```<startup>/<site-name>/local/resources.json```


```json
{
    "id": "sys_metrics_collector",
    "path": "nvflare.metrics.sys_metrics_collector.SysMetricsCollector",
    "args": {
        "tags": {
            "site": "<site>",
            "env": "dev"
        }
    }
}, 
{
    "id": "statsd_reporter",
    "path": "nvflare.fuel_opt.statsd.statsd_reporter.StatsDReporter",
    "args": {
        "host": "<statsd_exporter_host>",
        "port": <statsd_exporter_port>
    }
},
{
    "id": "remote_metrics_receiver",
    "path": "nvflare.metrics.remote_metrics_reciever.RemoteMetricsReceiver",
     "args": {
         "events": ["fed.metrics_event"]
     }
}
```


#### 3. Individual Monitoring Systems for Each Site

The configuration for this setup should be the same as setup 1, the only differences are the statsd-exporters' hosts and ports are not the same. 


### Summary

By following the steps outlined above, you can set up NVFLARE to publish metrics to StatsD Exporter, Prometheus, and Grafana, using one of the three setup types that best fits your needs.

## Job Example

We are are going to use hello-pt as example and demonstrate the two setups scenarios. Please continue to with [Job Example](jobs/README.md)



