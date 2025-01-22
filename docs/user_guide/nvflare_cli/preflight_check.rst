.. _preflight_check:

****************************************
NVIDIA FLARE Preflight Check
****************************************

The NVIDIA FLARE preflight check was added in version 2.2 to help perform preliminary checks before users start an
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


Preflight Check for Overseer
============================

If the overseer package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00" and it is called "overseer",
on the overseer site, a user should run: 

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/overseer

If things passed, users will see things like:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer port binding,PASSED,N/A.
    Check dry run,	PASSED,N/A.

If something is wrong, it will be reported:
The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer port binding,Can't bind to address ({address}) for overseer service: {e},Please check the DNS and port.
    Check dry run,	Can't start successfully,	Please check the error message of dry run.


Server
======

Before you check the server, make sure the overseer is running.

If the server package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00" and it is called "server1",
on the server site, a user should run: 

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/server1

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check overseer running,	Can't connect to overseer,"1) Please check if overseer is up or certificates are correct
    2) Please check if overseer hostname in project.yml is available
    3) If running in local machine, check if overseer defined in project.yml is defined in /etc/hosts."
    Check grpc port binding,Can't bind to address ({grpc_target_address}) for grpc service: {e},Please check the DNS and port.
    Check admin port binding,Can't bind to address ({admin_host}:{admin_port}) for admin service: {e},Please check the DNS and port.
    Check snapshot storage writable,Can't write to {self.snapshot_storage_root}: {e}.,Please check the user permission.
    Check job storage writable,	Can't write to {self.job_storage_root}: {e}.,Please check the user permission.
    Check dry run,Can't start successfully: {error},Please check the error message of dry run.

similarly, you can check server2 if you are in the HA mode

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/server2



Client
======

Before you check the clients, make sure the overseer and server are running.

If the client package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00" and it is called "site-1"
So on the client site, a user will run: 

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/site-1

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check overseer running,	Can't connect to overseer,"1) Please check if overseer is up or certificates are correct
    2) Please check if overseer hostname in project.yml is available
    3) If running in local machine, check if overseer defined in project.yml is defined in /etc/hosts."
    Check primary service provider available,Can't get primary service provider ({psp}) from overseer,Please check if server is up.
    Check SP's socket server available,Can't connect to primary service provider's ({sp_end_point}) socketserver,Please check if server is up.
    Check SP's GRPC server available,Can't connect to primary service provider's ({sp_end_point}) grpc server,Please check if server is up.
    Check dry run,	Can't start successfully: {error},	Please check the error message of dry run.


Admin
=====

Before you check the FLARE Console (previously called the admin client before version 2.2), make sure the overseer and server are running.

If the FLARE Console package is in "/path_to_NVFlare/NVFlare/workspace/example_project/prod_00/" and it is called "admin@nvidia.com",
a user should run:

.. code-block::

  nvflare preflight_check -p /path_to_NVFlare/NVFlare/workspace/example_project/prod_00/admin@nvidia.com

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 25

    Check overseer running,	Can't connect to overseer,"1) Please check if overseer is up or certificates are correct
    2) Please check if overseer hostname in project.yml is available
    3) if running on a local machine, check if the overseer defined in project.yml is defined in /etc/hosts."
    Check primary service provider available,Can't get primary service provider ({psp}) from overseer,Please check if server is up.
    Check SP's socket server available,Can't connect to primary service provider's ({sp_end_point}) socketserver,Please check if server is up.
    Check SP's GRPC server available,Can't connect to primary service provider's ({sp_end_point}) grpc server,Please check if server is up.
    Check dry run,	Can't start successfully: {error},	Please check the error message of dry run.
