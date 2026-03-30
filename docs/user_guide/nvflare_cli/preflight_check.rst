.. _preflight_check:

****************************************
NVIDIA FLARE Preflight Check
****************************************

The NVIDIA FLARE preflight check is to help perform preliminary checks before users start an
NVFlare subsystem on their machine to catch errors early and mitigate the pain of setting up and running jobs in
NVIDIA FLARE.

General Usage
=============

.. code-block::

    nvflare preflight_check -p PACKAGE_PATH
    nvflare preflight_check --package_path PACKAGE_PATH


This preflight check script should be run on each site's machine. The ``PACKAGE_PATH`` is the path to the folder that contains
the package to be checked.

After running the script, for the checks that pass, users will see "PASSED". The problem and how
to fix it is reported for checks that fail.

Below are the scripts to run the preflight check on each type of site and the possible problems that may be reported.


Preflight check on server site
------------------------------

If the server package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00" and it is called "server1",
on the server site, a user should run: 

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/server1

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check grpc port binding,Can't bind to address ({grpc_target_address}) for grpc service: {e},Please check the DNS and port.
    Check admin port binding,Can't bind to address ({admin_host}:{admin_port}) for admin service: {e},Please check the DNS and port.
    Check snapshot storage writable,Can't write to {self.snapshot_storage_root}: {e}.,Please check the user permission.
    Check job storage writable,	Can't write to {self.job_storage_root}: {e}.,Please check the user permission.
    Check dry run,Can't start successfully: {error},Please check the error message of dry run.


Preflight check on client sites
-------------------------------

Before you check the clients, make sure the server is running.

If the client package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00" and it is called "site-1"
So on the client site, a user will run: 

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/site-1

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check GRPC server available,Can't connect to grpc ({server_name}:{grpc_port}) server,Please check if server is up.
    Check dry run,	Can't start successfully: {error},	Please check the error message of dry run.


Preflight check for admin consoles 
----------------------------------

Before you check the FLARE Admin Console, make sure the server is running.

If the FLARE Console package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00/" and it is called "admin@nvidia.com",
a user should run:

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/admin@nvidia.com

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check GRPC server available,Can't connect to grpc ({server_name}:{grpc_port}) server,Please check if server is up.
    Check dry run,	Can't start successfully: {error},	Please check the error message of dry run.
