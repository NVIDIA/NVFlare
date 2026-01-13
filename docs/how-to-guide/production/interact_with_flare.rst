.. _interact_with_flare_guide:

#################################
How to Interact with FLARE System
#################################

This guide covers two primary methods for interacting with a running NVIDIA FLARE system:

1. **FLARE API** - Python API for programmatic control from scripts or Jupyter notebooks
2. **Admin Console** - Interactive command-line interface for system management


FLARE API (Python)
==================

The FLARE API provides programmatic access to the FL system, making it ideal for:

- Jupyter notebook workflows
- Automated job pipelines
- Custom monitoring scripts
- Integration with MLOps tools

Getting Started
---------------

Initialize a secure session using your admin startup kit:

.. code-block:: python

    from nvflare.fuel.flare_api.flare_api import new_secure_session

    # Connect to the FL system
    sess = new_secure_session(
        username="admin@nvidia.com",
        startup_kit_dir="/path/to/admin@nvidia.com"
    )

The session automatically handles authentication using the certificates in your startup kit.

Basic Operations
----------------

**Get System Information**

.. code-block:: python

    # Get system status and connected clients
    system_info = sess.get_system_info()
    print(system_info)

**Submit a Job**

.. code-block:: python

    # Submit a job and get the job ID
    job_id = sess.submit_job("/path/to/job_folder")
    print(f"Submitted job: {job_id}")

**Monitor Job Progress**

.. code-block:: python

    # Wait for job to complete
    result = sess.monitor_job(job_id)
    print(f"Job completed with status: {result}")

**List and Manage Jobs**

.. code-block:: python

    # List all jobs
    jobs = sess.list_jobs()
    print(jobs)

    # Get job metadata
    job_meta = sess.get_job_meta(job_id)
    print(job_meta)

    # Clone a job
    new_job_id = sess.clone_job(job_id)

    # Abort a running job
    sess.abort_job(job_id)

    # Download job results
    sess.download_job(job_id)

Complete Workflow Example
-------------------------

Here's a complete example for a Jupyter notebook:

.. code-block:: python

    from nvflare.fuel.flare_api.flare_api import new_secure_session

    # Initialize session
    sess = new_secure_session(
        "admin@nvidia.com",
        "/workspace/project/prod_00/admin@nvidia.com"
    )

    try:
        # Check system status
        print("System Info:")
        print(sess.get_system_info())

        # Submit job
        job_id = sess.submit_job("/workspace/jobs/my_fl_job")
        print(f"\nSubmitted job: {job_id}")

        # Monitor until completion
        print("\nMonitoring job progress...")
        result = sess.monitor_job(job_id, poll_interval=5.0)
        print(f"Job finished: {result}")

        # Download results
        sess.download_job(job_id)
        print(f"Results downloaded to current directory")

    finally:
        # Always close the session
        sess.close()

Custom Job Monitoring
---------------------

For advanced monitoring, provide a custom callback:

.. code-block:: python

    def my_monitor_callback(session, job_id, job_meta, *args, **kwargs):
        """Custom callback called after each status poll."""
        status = job_meta.get("status", "UNKNOWN")
        print(f"Job {job_id}: {status}")

        # Return True to continue monitoring, False to stop
        if status == "RUNNING":
            return True
        return True  # Continue until job completes

    # Monitor with custom callback
    sess.monitor_job(
        job_id,
        timeout=3600,        # 1 hour timeout
        poll_interval=10.0,  # Check every 10 seconds
        cb=my_monitor_callback
    )

Session Management
------------------

Always close the session when done:

.. code-block:: python

    # Using try/finally
    try:
        # ... your code ...
    finally:
        sess.close()

    # Or using context manager pattern
    sess = new_secure_session("admin@nvidia.com", "/path/to/startup")
    # ... use session ...
    sess.close()


Admin Console
=============

The Admin Console provides an interactive command-line interface for managing the FL system.

Starting the Console
--------------------

Launch the admin console from your startup kit:

.. code-block:: shell

    cd /path/to/admin@nvidia.com/startup
    ./fl_admin.sh

Enter your admin username when prompted (e.g., ``admin@nvidia.com``).

Getting Help
------------

.. code-block:: text

    > help                    # List all available commands
    > ? check_status          # Get help for a specific command

System Status Commands
----------------------

.. code-block:: text

    # Check server status and connected clients
    > check_status server

    # Check all client statuses
    > check_status client

    # Check specific client
    > check_status client site-1

    # Get system information
    > sys_info server
    > sys_info client site-1

Job Management Commands
-----------------------

.. code-block:: text

    # Submit a job
    > submit_job /path/to/job_folder

    # List all jobs
    > list_jobs

    # List jobs with name prefix
    > list_jobs -n my_experiment

    # Abort a running job
    > abort_job <job_id>

    # Clone a job (create a copy)
    > clone_job <job_id>

    # Download job results
    > download_job <job_id>

    # Delete a job from job store
    > delete_job <job_id>

File Operations
---------------

.. code-block:: text

    # List files on server
    > ls server

    # List files on client with details
    > ls site-1 -alt

    # View file contents
    > cat server startup/fed_server.json

    # Search in files
    > grep server "error" -i log.txt

    # View first/last lines
    > head server log.txt -n 20
    > tail site-1 log.txt -n 50

    # Print workspace directory
    > pwd server

System Control Commands
-----------------------

.. code-block:: text

    # Restart clients
    > restart client

    # Restart specific client
    > restart client site-1

    # Restart server (all clients also restart)
    > restart server

    # Shutdown clients
    > shutdown client

    # Shutdown server (shutdown clients first!)
    > shutdown server

    # Remove a disconnected client
    > remove_client site-1

Log Configuration
-----------------

.. code-block:: text

    # Configure server log level
    > configure_site_log server DEBUG

    # Configure client log level
    > configure_site_log client site-1 INFO

    # Configure job-specific logging
    > configure_job_log <job_id> server DEBUG

Output Redirection
------------------

Save command output to a file:

.. code-block:: text

    # Save and display output
    > sys_info server >server_info.txt

    # Save only (no display)
    > sys_info server >>server_info.txt

Exiting the Console
-------------------

.. code-block:: text

    > bye


Choosing Between FLARE API and Admin Console
============================================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Use Case
     - FLARE API
     - Admin Console
   * - Jupyter notebooks
     - ✓ Recommended
     -
   * - Automated pipelines
     - ✓ Recommended
     -
   * - Interactive debugging
     -
     - ✓ Recommended
   * - Quick status checks
     -
     - ✓ Recommended
   * - Custom monitoring
     - ✓ Recommended
     -
   * - File inspection
     -
     - ✓ Recommended
   * - MLOps integration
     - ✓ Recommended
     -


Troubleshooting
===============

**FLARE API connection fails**

- Verify the server is running
- Check the startup kit path is correct
- Ensure certificates haven't expired

**Admin console cannot connect**

- Verify the server is running and accessible
- Check network connectivity and firewall rules
- Ensure you're using the correct admin username

**Commands time out**

- Check network latency between admin and server
- Verify the server is not overloaded
- Try increasing timeout values


References
==========

- :ref:`flare_api` - Complete FLARE API documentation
- :ref:`operating_nvflare` - Full admin command reference
- :ref:`preflight_check` - System verification tool
