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

  nvflare preflight_check --package_root PACKAGE_ROOT --packages [PACKAGE_NAME1 PACKAGE_NAME2]

This preflight check script should be run on each site's machine. The ``package_root`` is the folder that contains
these packages.

After running the script, for the checks that pass, users will see "PASSED". The problem and how
to fix it is reported for checks that fail.

Below are the scripts to run the preflight check on each type of site and the possible problems that may be reported.


Preflight Check for Overseer
============================

If the overseer package is in "/home/user1" and it is called "overseer.example.com",
on the overseer site, a user should run: 

.. code-block::

  nvflare preflight_check --package_root /home/user1 --packages overseer.example.com

If things passed, users will see things like:


If something is wrong, it will be reported:



The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer port binding,Can't bind to address ({address}) for overseer service: {e},Please check the DNS and port.
    Check dry run,	Can't start successfully,	Please check the error message of dry run.


Server
======

If the server package is in "/home/user1" and it is called "example1.com",
on the server site, a user should run: 

.. code-block::

  nvflare preflight_check --package_root /home/user1 --packages example1.com

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer running,	Can't connect to overseer,Please check if overseer is up.
    Check grpc port binding,Can't bind to address ({grpc_target_address}) for grpc service: {e},Please check the DNS and port.
    Check admin port binding,Can't bind to address ({admin_host}:{admin_port}) for admin service: {e},Please check the DNS and port.
    Check snapshot storage writable,Can't write to {self.snapshot_storage_root}: {e}.,Please check the user permission.
    Check job storage writable,	Can't write to {self.job_storage_root}: {e}.,Please check the user permission.
    Check dry run,Can't start successfully: \n{out},Please check the error message of dry run.

Client
======

If the client package is in "/home/user1" and it is called "site-1"
So on the client site, a user will run: 

.. code-block::

  nvflare preflight_check --package_root /home/user1 --packages site-1

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer running,	Can't connect to overseer,	Please check if overseer is up.
    Check primary service provider available,Can't get primary service provider ({psp}) from overseer,Please check if server is up.
    Check SP's socket server available,Can't connect to primary service provider's ({sp_end_point}) socketserver,Please check if server is up.
    Check SP's GRPC server available,Can't connect to primary service provider's ({sp_end_point}) grpc server,Please check if server is up.
    Check dry run,	Can't start successfully: \n{out},	Please check the error message of dry run.


Admin
=====

If the FLARE Console (previously called the admin client before version 2.2) package is in "/home/user1" and it is called "admin@nvidia.com",
a user should run:

.. code-block::

  nvflare preflight_check --package_root /home/user1 --packages admin@nvidia.com

The problems that may be reported:

.. csv-table::
    :header: Checks,Problems,How to fix
    :widths: 15, 20, 15

    Check overseer running,	Can't connect to overseer,	Please check if overseer is up.
    Check primary service provider available,Can't get primary service provider ({psp}) from overseer,Please check if server is up.
    Check SP's socket server available,Can't connect to primary service provider's ({sp_end_point}) socketserver,Please check if server is up.
    Check SP's GRPC server available,Can't connect to primary service provider's ({sp_end_point}) grpc server,Please check if server is up.
    Check dry run,	Can't start successfully: \n{out},	Please check the error message of dry run.
