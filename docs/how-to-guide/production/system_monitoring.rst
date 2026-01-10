.. _sys_monitoring_guide:

###################################
How to Configure System Monitoring
###################################

This guide covers monitoring NVIDIA FLARE systems in production, including system metrics,
experiment tracking, logging, and auditing.

Overview
========

FLARE provides comprehensive monitoring capabilities across three areas:

1. **System Monitoring** - Job lifecycle, resource usage, network statistics
2. **Experiment Tracking** - Training metrics (loss, accuracy, etc.)
3. **Logging & Auditing** - System logs and security audit trails


System Monitoring with Prometheus/Grafana
=========================================

For production deployments, FLARE integrates with industry-standard monitoring tools
to provide real-time visibility into system operations.

Architecture
------------

.. code-block:: text

    FLARE System → StatsD Exporter → Prometheus → Grafana
                        ↓
                   System Metrics

Components:

- **StatsD Exporter**: Collects and exports system metrics
- **Prometheus**: Scrapes and stores metrics time-series data
- **Grafana**: Provides visualization dashboards

Metrics Collected
-----------------

- Job lifecycle events (start, complete, abort)
- System performance metrics
- Resource utilization
- Network statistics
- Component health status

Setup
-----

For detailed setup instructions, see the :github_nvflare_link:`Monitoring Example <examples/advanced/monitoring/README.md>`.

The basic steps are:

1. Install and configure StatsD Exporter
2. Set up Prometheus to scrape metrics
3. Install Grafana and import FLARE dashboards
4. Configure FLARE monitoring components


Experiment Tracking
===================

Track training metrics (loss, accuracy, AUC) across federated learning runs using
integrated experiment tracking tools.

Supported Tools
---------------

- **TensorBoard** - Visualization toolkit for ML experiments
- **MLflow** - ML lifecycle management platform
- **Weights & Biases** - Experiment tracking and model management

Client-Side Configuration
-------------------------

Add logging to your training code using FLARE's LogWriter APIs:

**TensorBoard:**

.. code-block:: python

    from nvflare.client.tracking import TBWriter

    tb_writer = TBWriter()
    tb_writer.add_scalar("loss", loss_value, global_step)

**MLflow:**

.. code-block:: python

    from nvflare.client.tracking import MLflowWriter

    mlflow = MLflowWriter()
    mlflow.log_metric("loss", loss_value, global_step)

**Weights & Biases:**

.. code-block:: python

    from nvflare.client.tracking import WandBWriter

    wandb = WandBWriter()
    wandb.log({"loss": loss_value}, step=global_step)

Streaming Metrics to Server
---------------------------

Configure clients to stream metrics to the server for centralized tracking.

Add to ``config_fed_client.json``:

.. code-block:: json

    {
        "id": "event_to_fed",
        "name": "ConvertToFedEvent",
        "args": {
            "events_to_convert": ["analytix_log_stats"],
            "fed_event_prefix": "fed."
        }
    }

Server-Side Receivers
---------------------

Configure the server to receive and record metrics.

**TensorBoard Receiver** (``config_fed_server.json``):

.. code-block:: json

    {
        "id": "tb_receiver",
        "name": "TBAnalyticsReceiver",
        "args": {"events": ["fed.analytix_log_stats"]}
    }

**MLflow Receiver**:

.. code-block:: json

    {
        "id": "mlflow_receiver",
        "name": "MLflowReceiver",
        "args": {
            "tracking_uri": "http://localhost:5000",
            "events": ["fed.analytix_log_stats"]
        }
    }

Start the MLflow server:

.. code-block:: shell

    mlflow server --host 0.0.0.0 --port 5000

For detailed examples, see :ref:`experiment_tracking_apis`.


Logging Configuration
=====================

FLARE uses Python's logging framework with configurable handlers and formatters.

Default Log Files
-----------------

FLARE generates multiple log files:

- ``log.txt`` - Main system log
- ``log_error.txt`` - Error-only log
- ``log.json`` - Structured JSON log
- ``log_fl.txt`` - FL-specific training log

Log Configuration File
----------------------

Customize logging by modifying ``log_config.json`` in your startup kit:

.. code-block:: json

    {
        "version": 1,
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "consoleFormatter"
            },
            "logFileHandler": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "filename": "log.txt",
                "maxBytes": 20971520,
                "backupCount": 10
            }
        },
        "loggers": {
            "root": {
                "level": "INFO",
                "handlers": ["consoleHandler", "logFileHandler"]
            }
        }
    }

Dynamic Log Configuration
-------------------------

Change log levels at runtime using admin commands:

.. code-block:: text

    # Configure server log level
    > configure_site_log server DEBUG

    # Configure client log level
    > configure_site_log client site-1 INFO

    # Configure job-specific logging
    > configure_job_log <job_id> server DEBUG

For detailed logging configuration, see :ref:`logging_configuration`.


Auditing
========

FLARE maintains audit logs for security and compliance, recording:

- User command events
- Job lifecycle events
- Authentication events
- Critical system events

Audit Log Locations
-------------------

Audit logs are stored in different locations:

- **Server Parent (SP)**: Server's root workspace directory
- **Server Job (SJ)**: Job's workspace directory
- **Client Parent (CP)**: Client's root workspace directory
- **Client Job (CJ)**: Job's workspace directory

Audit Log Format
----------------

Each line is an event with headers in square brackets:

.. code-block:: text

    [E:event-id][T:timestamp][U:user][A:action] message

Example:

.. code-block:: text

    [E:b6ac4a2a-eb01-4123-b898-758f20dc028d][T:2022-09-13 13:56:01.280558][U:admin@nvidia.com][A:submit_job] submitted job xyz

Headers:

- **E**: Event ID (unique identifier)
- **T**: Timestamp
- **U**: User who initiated the action
- **J**: Job ID (if applicable)
- **A**: Action performed
- **R**: Related event ID

Accessing Audit Logs
--------------------

Download job audit logs:

.. code-block:: text

    > download_job <job_id>

The downloaded job folder contains audit logs in the workspace directory.

For detailed auditing information, see :ref:`auditing`.


Statistics Pool Monitoring
==========================

FLARE's statistics pool system provides detailed metrics for monitoring communication patterns,
message timing, and system performance.

Pool Types
----------

- **Histogram Pools**: Track distributions of values (message sizes, timing) with configurable bins
- **Counter Pools**: Track simple counters for specific events

Common pools available:

- ``request_processing`` - Time spent processing requests
- ``request_response`` - End-to-end request-response times
- ``msg_sizes`` - Distribution of message sizes
- ``msg_travel_time`` - Message transmission times

Enabling Stats Pool Saving
--------------------------

Configure stats pool saving in your job's ``meta.json``:

.. code-block:: json

    {
        "name": "my_fl_job",
        "stats_pool_config": {
            "save_pools": ["*"]
        }
    }

Options for ``save_pools``:

- Specific pool names: ``["request_processing", "request_response"]``
- Wildcard for all pools: ``["*"]``
- Mix of both: ``["request_processing", "*"]``

Output Files
------------

After job completion, two files are generated in the job workspace:

**stats_pool_summary.json** - Aggregated statistics:

.. code-block:: json

    {
        "request_processing": {
            "name": "request_processing",
            "type": "hist",
            "description": "Request processing time",
            "bins": [
                {"range": "0-10ms", "count": 150, "avg": 5.2},
                {"range": "10-100ms", "count": 80, "avg": 45.3}
            ]
        }
    }

**stats_pool_records.csv** - Raw timestamped records:

.. code-block:: text

    timestamp,pool_name,value
    2024-01-15T10:30:45.123Z,request_processing,0.025
    2024-01-15T10:30:45.456Z,request_processing,0.031

Using the Stats Viewer Tool
---------------------------

FLARE provides a command-line tool to analyze statistics files:

.. code-block:: shell

    python -m nvflare.fuel.f3.qat.stats_viewer -f stats_pool_summary.json

Interactive commands:

.. code-block:: text

    > list_pools                     # List all available pools
    > show_pool request_processing   # Show pool with default mode
    > show_pool msg_sizes count      # Show with count mode
    > show_pool request_response avg # Show with average mode
    > bye                            # Exit viewer

Display modes: ``count``, ``total``, ``min``, ``max``, ``avg``

Runtime Pool Inspection
-----------------------

View statistics during job execution using admin commands:

.. code-block:: text

    # List active cells
    > cells

    # List pools on a cell
    > list_pools server.job_abc-123

    # Show pool statistics
    > show_pool server.job_abc-123 request_processing avg

    # Show message statistics
    > msg_stats server.job_abc-123

Analyzing Stats with Python
---------------------------

.. code-block:: python

    import json
    import pandas as pd

    # Load summary data
    with open('stats_pool_summary.json', 'r') as f:
        summary = json.load(f)

    # Load raw records
    records = pd.read_csv('stats_pool_records.csv')

    # Analyze timing patterns
    timing = records[records['pool_name'] == 'request_processing']
    print(f"Average: {timing['value'].mean():.3f}")
    print(f"95th percentile: {timing['value'].quantile(0.95):.3f}")
    print(f"Max: {timing['value'].max():.3f}")

For detailed information, see :ref:`diagnostic_commands`.


Diagnostic Commands
===================

FLARE provides diagnostic commands for troubleshooting system issues.

System Information
------------------

.. code-block:: text

    # Get server system info
    > sys_info server

    # Get client system info
    > sys_info client site-1

File Inspection
---------------

.. code-block:: text

    # View log files
    > tail server log.txt -n 50

    # Search for errors
    > grep server "ERROR" -i log.txt

    # List workspace files
    > ls server -alt

Cell Diagnostics
----------------

When diagnose mode is enabled:

.. code-block:: text

    # List active cells
    > cells

    # List statistics pools
    > list_pools server

    # Show pool statistics
    > show_pool server <pool_name>

    # Show message statistics
    > msg_stats server


Best Practices
==============

**Production Monitoring**

- Set up Prometheus/Grafana for real-time dashboards
- Configure alerts for job failures and resource issues
- Enable structured JSON logging for log aggregation tools

**Experiment Tracking**

- Use MLflow or W&B for centralized experiment management
- Stream metrics to server for cross-site comparison
- Tag experiments with meaningful identifiers

**Logging**

- Use INFO level for production, DEBUG for troubleshooting
- Enable log rotation to manage disk space
- Archive logs for compliance requirements

**Auditing**

- Regularly review audit logs for security incidents
- Back up audit logs for compliance
- Monitor for unauthorized access attempts


References
==========

- :ref:`monitoring` - System monitoring overview
- :ref:`diagnostic_commands` - Statistics pools and diagnostic commands
- :ref:`experiment_tracking_apis` - Experiment tracking APIs
- :ref:`logging_configuration` - Logging configuration guide
- :ref:`auditing` - Audit logging documentation
- :ref:`operating_nvflare` - Admin console commands
