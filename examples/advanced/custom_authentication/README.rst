Example of Custom Client Registration Authentication
======================================================


Overview
--------

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Demonstrate custom authentication policy

During the client registering to the server process, customer can build an event handler to listen to the EventType.CLIENT_REGISTERED event, and inject additional logic to verify and process the client registration. 
If a particular client does not meet certain criteria, the server can reject the client registration.
In this example, we create two clients, `site_a` and `site_b`. `site_b` is rejected by the server.

System Requirements
-------------------

1. Install Python and Virtual Environment,
::
    python3 -m venv nvflare-env
    source nvflare-env/bin/activate

2. Install NVFlare
::
    pip install -r requirements.txt

3. The example is part of the NVFlare source code. The source code can be obtained like this,
::
    git clone https://github.com/NVIDIA/NVFlare.git

4. TLS requires domain names. Please add following line in :code:`/etc/hosts` file,
::
    127.0.0.1	server1


Setup
-----

::
    cd NVFlare/examples/advanced/custom_authentication
    ./setup.sh

All the startup kits will be generated in this folder,
::
    /tmp/nvflare/poc/custom_authentication/prod_00

.. note::
   :code:`workspace` folder is removed everytime :code:`setup.sh` is run. Please do not save customized
   files in this folder.

Starting NVFlare
----------------

This script will start up the server and try to start 2 clients,
::
   nvflare poc start

As the server starts and client `site_b` tries to register to the server, the server will reject the client registration.
You will see the following error message in the log:

::
2026-01-28 11:47:48,667 - FederatedServer - ERROR - Failed to authenticate the register_client: NotAuthenticated: site_b not allowed to register

The terminal prompt will automatically go to the Admin Console. You can see there that only `site_a` was able to register to the server by typing:

::
    > check_status server

Output:

::

    Engine status: stopped
---------------------
| JOB_ID | APP NAME |
---------------------
---------------------
Registered clients: 1 
-----------------------------------------------------------------------------------------------------
| CLIENT | FQCN   | FQSN   | LEAF | TOKEN                                | LAST CONNECT TIME        |
-----------------------------------------------------------------------------------------------------
| site_a | site_a | site_a | True | b67de696-dca8-4c0a-885f-13eb9f94e9e1 | Wed Jan 28 11:48:30 2026 |
-----------------------------------------------------------------------------------------------------


Individual Server, Client, and Admin Startup
--------------------------------------------

Alternatively, you can start the server, client, and admin individually by running the following commands:

Start the server
~~~~~~~~~~~~~~~~

::

    cd /tmp/nvflare/poc/custom_authentication/prod_00/server1
    ./startup/start.sh


Participants
------------

Site_a
~~~~~~

::

    cd /tmp/nvflare/poc/custom_authentication/prod_00/site_a
    ./startup/start.sh

* site_a is able to start and register to the server.

Site_b
~~~~~~

::

    cd /tmp/nvflare/poc/custom_authentication/prod_00/site_b
    ./startup/start.sh

* site_b is NOT able to start and register to the server. It's blocked by the ServerCustomSecurityHandler logic during the client registration.

Logging with Admin Console
--------------------------

For example, this is how to login as :code:`super@a.org` user,

::

    cd /tmp/nvflare/poc/custom_authentication/prod_00/super@a.org
    ./startup/fl_admin.sh

At the prompt, enter the user email :code:`super@a.org`
