Example for Federated Policies
==============================


Overview
--------

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Demonstrate job-level authorization policy

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

The :code:`project.yml` file defines all the sites and users (called admin in NVFlare)
used in the examples. The startup kits will be created by :code:`setup.sh`
::
    cd NVFlare/examples/advanced/job-level-authorization
    ./setup.sh
All the startup kits will be generated in this folder,
::
    workspace/fed_policy/prod_00

.. note::
   :code:`workspace` folder is removed everytime :code:`setup.sh` is run. Please do not save customized
   files in this folder.

Starting NVFlare
________________

This script will start up the server and 2 clients,
::
   ./start.sh

Logging with Admin Console
__________________________

For example, this is how to login as :code:`admin@a.org` user,
::
    cd workspace/fed_policy/prod_00/admin@a.org
    ./startup/fl_admin.sh
At the prompt, enter the user email :code:`admin@a.org`

Multiple users can login at the same time by using multiple terminals.

The setup.sh has copied the jobs folder to the workspace folder.
So jobs can be submitted like this, type the following command in the admin console:

::
   submit_job ../../job1

Participants
------------
Site
____
* :code:`server1`: NVFlare server
* :code:`site_a`: Site_a has a CustomSecurityHandler set up which does not allow the job "FL Demo Job1" to run. Any other named jobs will be able to deploy and run on site_a.


Users
_____
* :code:`super@a.org`: Super user with role :code:`project_admin` who can do everything
* :code:`admin@a.org`: Admin for a.org with role :code:`org_admin`
* :code:`trainer@a.org`: Lead trainer for a.org with role :code:`lead`
* :code:`trainer@b.org`: Lead trainer for b.org with role :code:`lead`
* :code:`user@b.org`: Regular user for b.org with role :code:`member`

Jobs
____

* job1: The job is called  :code:`hello-numpy-sag`. site_a will allow this job to run.
* job2: The job is called  :code:`FL Demo Job1`. site_a will block this job to run.



Shutting down NVFlare
_____________________
All NVFlare server and clients can be stopped by using this script,
::
   ./stop.sh
