Example of Custom Client Registration Authentication
==============================


Overview
--------

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Demonstrate custom authentication policy

During the client registering to the server process, customer can build an event handler to listen to the EventType.CLIENT_REGISTERED event, and inject additional logic to verify and process the client registration. If a particular client does not meet certain criteria, the server can reject the client registration.

System Requirements
-------------------

1. Install Python and Virtual Environment,
::
    python3 -m venv nvflare-env
    source nvflare-env/bin/activate

2. Install NVFlare
::
    pip install nvflare

3. The example is part of the NVFlare source code. The source code can be obtained like this,
::
    git clone https://github.com/NVIDIA/NVFlare.git

4. TLS requires domain names. Please add following line in :code:`/etc/hosts` file,
::
    127.0.0.1	server1


Setup
_____

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
________________

This script will start up the server and 2 clients,
::
   nvflare poc start

Logging with Admin Console
__________________________

For example, this is how to login as :code:`super@a.org` user,
::
    cd /tmp/nvflare/poc/custom_authentication/prod_00/super@a.org
    ./startup/fl_admin.sh
At the prompt, enter the user email :code:`super@a.org`

Start the server
________________
    cd /tmp/nvflare/poc/custom_authentication/prod_00/server1
    ./startup/start.sh


Participants
------------
Site_a
____
    cd /tmp/nvflare/poc/custom_authentication/prod_00/site_a
    ./startup/start.sh
* site_a is able to start and register to the server.

Site_b
____
    cd /tmp/nvflare/poc/custom_authentication/prod_00/site_a
    ./startup/start.sh
* site_b is NOT able to start and register to the server. It's blocked by the ServerCustomSecurityHandler logic during the client registration.
