.. _diagnostic_commands:

#####################
Diagnostic Commands
#####################

NVIDIA FLARE provides diagnostic commands for monitoring and debugging communication statistics in the CellNet layer. These commands are particularly useful for troubleshooting network issues, analyzing message patterns, and understanding system performance characteristics.

.. note::
   These diagnostic commands are only available when the system is configured with diagnose mode enabled in the NetManager component.

Overview
========

The diagnostic commands allow administrators to:

* Discover active cells in the CellNet system
* View statistics about message sizes and timing
* Monitor communication patterns between cells
* Inspect available statistics pools
* Analyze histogram data with different statistical modes

These commands query the CellNet layer's statistics tracking system, which maintains various statistics pools for monitoring different aspects of system communication.

Statistics Pools
================

NVIDIA FLARE's statistics system uses "pools" to organize different types of metrics:

* **Histogram Pools**: Track distributions of values (e.g., message sizes, timing) with configurable bins
* **Counter Pools**: Track simple counters for specific events

Each pool has a name, type, and description. The system automatically creates pools for tracking message statistics, and applications can create custom pools for tracking domain-specific metrics.

Configuring Statistics Pool Saving
===================================

By default, statistics pools are maintained in memory during job execution. However, you can configure NVFLARE to save pool statistics to disk for later analysis and record-keeping.

Configuration in meta.json
---------------------------

To enable statistics pool saving for a job, add the following configuration to your job's ``meta.json`` file:

.. code-block:: json

   {
     "stats_pool_config": {
       "save_pools": [
         "request_processing",
         "request_response",
         "*"
       ]
     }
   }

**Configuration Options:**

* ``save_pools``: A list of pool names to save. Supports:
  
  * **Specific pool names**: e.g., ``"request_processing"``, ``"msg_sizes"``
  * **Wildcard**: Use ``"*"`` to save all pools
  * **Mixed**: Combine specific names and wildcards

**Examples:**

1. Save only specific pools:

.. code-block:: json

   {
     "stats_pool_config": {
       "save_pools": ["request_processing", "request_response"]
     }
   }

2. Save all pools:

.. code-block:: json

   {
     "stats_pool_config": {
       "save_pools": ["*"]
     }
   }

Output Files
------------

When statistics pool saving is enabled, NVFLARE generates two files for each job at the end of job execution:

stats_pool_summary.json
^^^^^^^^^^^^^^^^^^^^^^^^

**Location:** Job workspace directory

**Content:** Contains histogram summaries and aggregated statistics for each saved pool.

**Format:** JSON file with the following structure:

.. code-block:: json

   {
     "pool_name": {
       "name": "pool_name",
       "type": "hist",
       "description": "Pool description",
       "marks": [0, 10, 100, 1000],
       "bins": [
         {"range": "0-10", "count": 150, "total": 750.5, "min": 0.1, "max": 9.9},
         {"range": "10-100", "count": 80, "total": 4200.0, "min": 10.2, "max": 99.8},
         {"range": "100-1000", "count": 20, "total": 12000.0, "min": 105.0, "max": 950.0}
       ]
     },
     "another_pool": {
       ...
     }
   }

**Use Cases:**

* Post-job analysis of communication patterns
* Historical comparison across multiple job runs
* Generating reports for system performance
* Identifying trends over time

stats_pool_records.csv
^^^^^^^^^^^^^^^^^^^^^^^

**Location:** Job workspace directory

**Content:** Contains raw, timestamped recordings for each data point collected in the saved pools.

**Format:** CSV file with columns that vary by pool type:

For histogram pools:

.. code-block:: text

   timestamp,pool_name,value,additional_metadata
   2024-01-15T10:30:45.123Z,request_processing,0.025,
   2024-01-15T10:30:45.456Z,request_processing,0.031,
   2024-01-15T10:30:46.789Z,request_response,0.120,

For counter pools:

.. code-block:: text

   timestamp,pool_name,counter_name,value
   2024-01-15T10:30:45.123Z,event_counts,task_received,1
   2024-01-15T10:30:46.456Z,event_counts,task_completed,1

**Use Cases:**

* Detailed timeline analysis
* Custom data processing and visualization
* Integration with external analytics tools
* Machine learning on system behavior patterns
* Debugging specific events or anomalies

Workflow Example
----------------

**Step 1: Configure Your Job**

Create or edit your job's ``meta.json``:

.. code-block:: json

   {
     "name": "my_federated_job",
     "resource_spec": {},
     "min_clients": 2,
     "stats_pool_config": {
       "save_pools": ["*"]
     }
   }

**Step 2: Submit the Job**

.. code-block:: shell

   > submit_job my_job_folder

**Step 3: Run the Job**

The job executes normally, with statistics being collected in the background.

**Step 4: Retrieve Statistics After Job Completion**

.. code-block:: shell

   > download_job job_abc-123-def

**Step 5: Analyze the Output**

Navigate to the downloaded job workspace and examine:

.. code-block:: shell

   cd downloaded_job/workspace/
   
   # View summary statistics
   cat stats_pool_summary.json
   
   # Analyze raw records
   cat stats_pool_records.csv

**Step 6: Use Statistics for Analysis**

.. code-block:: python

   import json
   import pandas as pd
   
   # Load summary data
   with open('stats_pool_summary.json', 'r') as f:
       summary = json.load(f)
   
   # Load raw records
   records = pd.read_csv('stats_pool_records.csv')
   
   # Analyze timing patterns
   timing_data = records[records['pool_name'] == 'request_processing']
   print(f"Average: {timing_data['value'].mean()}")
   print(f"95th percentile: {timing_data['value'].quantile(0.95)}")

Using the Stats Viewer Tool
----------------------------

NVFLARE provides a convenient command-line tool called ``stats_viewer`` for interactively exploring statistics files. This tool allows you to view and analyze the ``stats_pool_summary.json`` files without writing custom scripts.

**Starting the Stats Viewer:**

.. code-block:: shell

   python -m nvflare.fuel.f3.qat.stats_viewer -f stats_pool_summary.json

This launches an interactive shell where you can explore the statistics data.

**Available Commands:**

The stats viewer provides the following commands:

* ``list_pools``: Display all available statistics pools with their types and descriptions
* ``show_pool <pool_name> [mode]``: Display detailed statistics for a specific pool
  
  * ``pool_name``: Name of the pool to display
  * ``mode`` (optional): Histogram display mode - one of: ``count``, ``total``, ``min``, ``max``, ``avg``

* ``help`` or ``?``: List available commands
* ``bye``: Exit the stats viewer

**Example Session:**

.. code-block:: shell

   $ python -m nvflare.fuel.f3.qat.stats_viewer -f stats_pool_summary.json
   Type help or ? to list commands.
   
   > list_pools
   Name                  Type    Description
   -------------------- ------- ------------------------------
   request_processing   hist    Request processing time
   request_response     hist    Request-response round trip
   msg_sizes            hist    Message size distribution
   
   > show_pool request_processing avg
   Range         Count    Average
   ------------ ------- -----------
   0-10ms           150     5.2ms
   10-100ms          80    45.3ms
   100-1000ms        20   425.8ms
   
   > show_pool msg_sizes count
   Range         Count
   ------------ -------
   0-1KB           200
   1KB-10KB        150
   10KB-100KB       50
   
   > bye

**Server-Side vs Client-Side Statistics:**

The ``stats_viewer`` tool can analyze statistics from both server and client sides:

* **Server-side statistics**: Available in the server's job workspace after job completion. Can be retrieved using ``download_job`` command.
* **Client-side statistics**: Currently stored locally on each client site in their respective job workspaces.

.. note::
   Currently, client-side statistics files are not automatically sent to the server after job completion. To analyze client statistics, you need to access the ``stats_pool_summary.json`` file directly on each client site's job workspace.

Common Pool Names
-----------------

The following pools are commonly available in NVFLARE jobs:

Communication Pools
^^^^^^^^^^^^^^^^^^^

* ``request_processing``: Time spent processing requests
* ``request_response``: End-to-end request-response times
* ``msg_sizes``: Distribution of message sizes
* ``msg_travel_time``: Message transmission times

Job-Specific Pools
^^^^^^^^^^^^^^^^^^

Different jobs may create custom pools based on their workflows. Use the ``list_pools`` command during job execution to discover available pools:

.. code-block:: shell

   > cells
   server.job_abc-123
   site1.job_abc-123
   
   > list_pools server.job_abc-123


Integration with Monitoring
----------------------------

Statistics pool data complements external monitoring systems:

* **Statistics Pools**: Detailed, job-specific metrics saved with job artifacts
* **External Monitoring** (Prometheus/Grafana): Real-time system-wide monitoring

Use both approaches together:

1. External monitoring for real-time alerting and dashboards
2. Statistics pool saving for detailed post-job analysis and historical records

See :ref:`monitoring` for information on setting up external monitoring.

Available Commands
==================

cells
-----

**Description:** Lists all active cells in the CellNet system with their FQCNs (Fully Qualified Cell Names). This command is essential for discovering available targets to use with other diagnostic commands.

**Usage:**

.. code-block:: shell

   cells

**Parameters:**

None. This command takes no parameters.

**Output:**

Displays a list of all active cells in the system, showing each cell's FQCN on a separate line, followed by a summary line showing the total number of valid cells.

**Example:**

.. code-block:: shell

   > cells

**Example Output:**

.. code-block:: text

   server
   site1
   site2
   site3
   server.abc-123-def
   site1.abc-123-def
   site2.abc-123-def
   Total Cells: 7

**Understanding the Output:**

The cells listed include:

* **Parent Cells**: Base cells for each site (e.g., ``server``, ``site1``, ``site2``)
  
  * The server's parent cell is always named ``server``
  * Client parent cells use their site names

* **Job Cells**: Cells created for active jobs (e.g., ``server.abc-123-def``, ``site1.abc-123-def``)
  
  * Format: ``<site_name>.<job_id>``
  * Created when a job is deployed
  * Removed when the job completes

* **Relay Cells**: In hierarchical deployments, relay nodes (e.g., ``relay1``, ``relay1.site1``)
  
  * Intermediate nodes in the communication hierarchy
  * Can have their own job cells when jobs are running

**Use Cases:**

* **Discover Available Targets**: Find valid FQCNs to use with ``list_pools``, ``show_pool``, ``msg_stats``, and other diagnostic commands
* **Verify System Topology**: Confirm all expected sites are connected and active
* **Monitor Job Cells**: See which jobs are currently running by identifying job cell FQCNs
* **Troubleshoot Connectivity**: Identify missing or disconnected cells
* **Understand Hierarchy**: In hierarchical deployments, visualize the cell structure

**Examples with Follow-up Commands:**

After running ``cells`` to discover targets, you can use the FQCNs with other commands:

.. code-block:: shell

   # First, discover all cells
   > cells
   server
   site1
   site2
   server.job123
   site1.job123
   Total Cells: 5

   # Then query specific cells
   > msg_stats server
   > msg_stats site1
   > msg_stats server.job123
   > list_pools site1.job123

**Interpreting Different Cell Types:**

1. **Server Parent Cell** (``server``):
   
   * Always present when the FL system is running
   * Handles administrative operations
   * Parent for all job cells on the server

2. **Client Parent Cells** (``site1``, ``site2``, etc.):
   
   * One per connected FL client site
   * Active as long as the client is connected
   * Persist across multiple jobs

3. **Job Server Cell** (``server.<job_id>``):
   
   * Created when a job is deployed on the server
   * Contains job-specific server workflows
   * Removed when job completes

4. **Job Client Cells** (``<site_name>.<job_id>``):
   
   * One per client participating in a job
   * Execute the client-side job logic
   * Communication with corresponding server job cell

5. **Hierarchical Cells** (``relay1``, ``relay1.site1``):
   
   * Relay nodes in hierarchical deployments
   * Can be nested (e.g., ``relay1.relay2.site1``)
   * Help manage large-scale deployments

**Tips:**

* Run ``cells`` before other diagnostic commands to identify valid targets
* Compare cell lists over time to track system changes
* If expected cells are missing, check connectivity and site status
* Job cells appear when jobs start and disappear when they complete

list_pools
----------

**Description:** Lists all statistics pools available on a target cell.

**Usage:**

.. code-block:: shell

   list_pools target

**Parameters:**

* ``target`` - The FQCN (Fully Qualified Cell Name) of the target cell to query (e.g., "server", "client1", "server.job_id")

**Output:**

Displays a table with three columns:

* **pool** - The name of the statistics pool
* **type** - The type of pool ("hist" for histogram, "counter" for counter)
* **description** - A description of what the pool tracks

**Example:**

.. code-block:: shell

   > list_pools server

**Example Output:**

.. code-block:: text

   +------------------+----------+--------------------------------+
   | pool             | type     | description                    |
   +------------------+----------+--------------------------------+
   | msg_travel_time  | hist     | Message travel time in seconds |
   | msg_sizes        | hist     | Message size distribution      |
   | request_counts   | counter  | Request counts by channel      |
   +------------------+----------+--------------------------------+

**Use Cases:**

* Discover available statistics pools on a cell
* Verify that expected statistics tracking is configured
* Identify pools for detailed inspection with ``show_pool``

show_pool
---------

**Description:** Shows detailed statistics for a specific pool on a target cell.

**Usage:**

.. code-block:: shell

   show_pool target pool_name [mode]

**Parameters:**

* ``target`` - The FQCN of the target cell to query
* ``pool_name`` - The name of the statistics pool to display
* ``mode`` - (Optional) The display mode for histogram pools. Valid values:

  * ``count`` - Show the count of values in each bin (default)
  * ``percent`` - Show the percentage of values in each bin
  * ``avg`` - Show the average value in each bin
  * ``min`` - Show the minimum value in each bin
  * ``max`` - Show the maximum value in each bin

**Output:**

For histogram pools, displays a table showing the distribution of values across bins. The exact columns depend on the pool type and configuration.

For counter pools, displays a table with counter names and their current values.

**Examples:**

.. code-block:: shell

   # Show message size distribution with counts
   > show_pool server msg_sizes count

   # Show message timing with averages
   > show_pool server msg_travel_time avg

   # Show message size percentages
   > show_pool site1 msg_sizes percent

**Example Output (Count Mode):**

.. code-block:: text

   +---------------+-------+
   | Range         | Count |
   +---------------+-------+
   | 0-1KB         | 150   |
   | 1KB-10KB      | 450   |
   | 10KB-100KB    | 80    |
   | 100KB-1MB     | 20    |
   | >1MB          | 5     |
   +---------------+-------+

**Example Output (Average Mode):**

.. code-block:: text

   +---------------+-----------+
   | Range         | Avg (sec) |
   +---------------+-----------+
   | 0-10ms        | 5.2e-03   |
   | 10ms-100ms    | 4.5e-02   |
   | 100ms-1s      | 3.2e-01   |
   | >1s           | 2.1e+00   |
   +---------------+-----------+

**Use Cases:**

* Analyze message size distributions to identify outliers
* Monitor timing characteristics of requests
* Compare statistics across different cells
* Identify performance bottlenecks or unusual patterns

msg_stats
---------

**Description:** Shows message request statistics for a target cell. This is a convenience command that displays the pre-configured message statistics pool.

**Usage:**

.. code-block:: shell

   msg_stats target [mode]

**Parameters:**

* ``target`` - The FQCN of the target cell to query
* ``mode`` - (Optional) The display mode. Valid values:

  * ``count`` - Show the count of messages (default)
  * ``percent`` - Show the percentage of messages
  * ``avg`` - Show the average message size or timing
  * ``min`` - Show the minimum values
  * ``max`` - Show the maximum values

**Output:**

Displays statistics about request messages, typically showing distributions of message sizes and/or timing information. The exact format depends on how the message statistics pool is configured in the system.

**Examples:**

.. code-block:: shell

   # Show message counts
   > msg_stats server

   # Show average message characteristics
   > msg_stats server avg

   # Show maximum values
   > msg_stats client1 max

**Example Output:**

.. code-block:: text

   Message Statistics for server:
   +---------------+-------+----------+
   | Size Range    | Count | Avg Time |
   +---------------+-------+----------+
   | 0-1KB         | 245   | 12ms     |
   | 1KB-10KB      | 180   | 25ms     |
   | 10KB-100KB    | 45    | 150ms    |
   | >100KB        | 10    | 500ms    |
   +---------------+-------+----------+

**Use Cases:**

* Quick overview of message traffic patterns
* Monitor communication health
* Identify unusual message patterns
* Baseline system performance characteristics

Common Workflows
================

Discovering Available Targets
------------------------------

Before using diagnostic commands, discover available cells:

1. **List all active cells:**

   .. code-block:: shell

      > cells

2. **Identify target cells of interest:**

   * Parent cells for overall system monitoring (``server``, ``site1``, etc.)
   * Job cells for job-specific monitoring (``server.job_id``, ``site1.job_id``)
   * Relay cells in hierarchical deployments

3. **Verify cell connectivity:**

   Check that expected cells appear in the list. Missing cells may indicate connectivity issues.

Investigating Communication Issues
----------------------------------

When investigating communication problems between cells:

1. **Discover active cells:**

   .. code-block:: shell

      > cells

2. **List available pools:**

   .. code-block:: shell

      > list_pools server
      > list_pools client1

3. **Check message statistics:**

   .. code-block:: shell

      > msg_stats server count
      > msg_stats client1 count

4. **Examine specific pools:**

   .. code-block:: shell

      > show_pool server msg_travel_time avg
      > show_pool client1 msg_sizes percent

Performance Analysis
--------------------

To analyze system performance characteristics:

1. **Check message timing distribution:**

   .. code-block:: shell

      > show_pool server msg_travel_time count
      > show_pool server msg_travel_time avg

2. **Analyze message size patterns:**

   .. code-block:: shell

      > show_pool server msg_sizes count
      > show_pool server msg_sizes max

3. **Compare across cells:**

   .. code-block:: shell

      > msg_stats server avg
      > msg_stats client1 avg
      > msg_stats client2 avg

Monitoring Job Execution
-------------------------

During job execution, monitor communication patterns:

1. **Identify job cells:**

   .. code-block:: shell

      > cells
      # Look for cells with format: <site_name>.<job_id>

2. **Check job cell statistics:**

   .. code-block:: shell

      > list_pools server.job_abc123
      > msg_stats server.job_abc123 count

3. **Compare parent and job cells:**

   .. code-block:: shell

      > msg_stats server avg
      > msg_stats server.job_abc123 avg

Statistical Modes Explained
============================

The different statistical modes provide different views of the data:

count
-----
Shows the number of data points in each bin. This is useful for understanding the distribution and identifying where most values fall.

**Use case:** "How many messages are in the 1KB-10KB range?"

percent
-------
Shows what percentage of all data points fall in each bin. This normalizes the distribution and makes it easier to compare across different time periods or cells.

**Use case:** "What percentage of messages are larger than 100KB?"

avg
---
Shows the average value of data points within each bin. This helps understand the typical characteristics within each range.

**Use case:** "For messages in the 10ms-100ms latency range, what's the typical latency?"

min
---
Shows the minimum value encountered in each bin. Useful for understanding best-case scenarios.

**Use case:** "What's the fastest response time we've seen in the 1KB-10KB message range?"

max
---
Shows the maximum value encountered in each bin. Useful for identifying worst-case scenarios or outliers.

**Use case:** "What's the longest latency we've seen for small messages?"

Target Cell Addressing
======================

The ``target`` parameter in these commands uses FQCN (Fully Qualified Cell Name) addressing:

Server Cell
-----------

.. code-block:: shell

   > msg_stats server

Client Cell
-----------

.. code-block:: shell

   > msg_stats site1
   > msg_stats client_alpha

Job Cells
---------

When a job is running, each site has a dedicated job cell with FQCN in the format ``<site_name>.<job_id>``:

.. code-block:: shell

   > msg_stats server.abc-123-def
   > msg_stats site1.abc-123-def

Hierarchical Cells
------------------

In hierarchical deployments with relays:

.. code-block:: shell

   > msg_stats relay1
   > msg_stats relay1.site1

See :ref:`hierarchical_communication` for more information on communication hierarchies.

Tips and Best Practices
========================

1. **Regular Monitoring:** Establish baseline statistics during normal operation to help identify anomalies.

2. **Compare Cells:** Compare statistics across different cells to identify inconsistencies or issues specific to certain sites.

3. **Use Different Modes:** Switch between statistical modes to get different insights into the same data.

4. **Track Over Time:** Run commands periodically and save output to track trends over time.

5. **Job-Specific Analysis:** Monitor job cells separately from parent cells to understand job-specific communication patterns.

6. **Correlate with Logs:** Use diagnostic commands in conjunction with log analysis for comprehensive troubleshooting.

Troubleshooting
===============

Command Not Found
-----------------

If diagnostic commands are not available:

* Verify that the NetManager component is configured with ``diagnose=True``
* Check that you have appropriate permissions to run these commands
* Ensure you're using a version of NVIDIA FLARE that includes these commands

Cells Command Shows Fewer Cells Than Expected
----------------------------------------------

If the ``cells`` command doesn't show all expected cells:

* **Check connectivity**: Verify that all sites are connected to the server
* **Check site status**: Use ``check_status`` to see if clients are properly connected
* **Wait for initialization**: Sites may take a few moments to appear after starting
* **Check logs**: Review server and client logs for connection errors
* **Verify network**: Ensure there are no network issues or firewall blocks

Cells Command Shows Old Job Cells
----------------------------------

If job cells remain listed after a job completes:

* There may be a delay in cleanup - wait a few moments and run ``cells`` again
* Check if the job is actually still running with ``list_jobs``
* Review logs for any errors during job shutdown

Invalid Mode Error
------------------

If you receive an "invalid mode" error:

* Ensure you're using one of the valid modes: ``count``, ``percent``, ``avg``, ``min``, ``max``
* Check for typos in the mode parameter
* Note that mode is case-sensitive (use lowercase)

Target Not Found
----------------

If the target cell cannot be reached:

* Verify the FQCN is correct
* Check that the target cell is running and connected
* Use the ``cells`` command to list available cells

Pool Does Not Exist
-------------------

If you receive a "pool does not exist" error:

* Use ``list_pools`` to see available pools on that cell
* Verify the pool name is spelled correctly
* Note that pool names are case-sensitive

See Also
========

* :ref:`cellnet_architecture` - Learn about FLARE's communication layer
* :ref:`communication_configuration` - Configure communication settings
* :ref:`monitoring` - Set up external monitoring with Prometheus and Grafana
* :ref:`hierarchical_communication` - Understand hierarchical cell topologies

