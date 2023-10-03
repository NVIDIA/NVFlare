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

::
    cd NVFlare/examples/advanced/job-level-authorization
    ./setup.sh
All the startup kits will be generated in this folder,
::
    /tmp/nvflare/poc/job-level-authorization/prod_00

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
    cd /tmp/nvflare/poc/job-level-authorization/prod_00/super@a.org
    ./startup/fl_admin.sh
At the prompt, enter the user email :code:`super@a.org`

The setup.sh has copied the jobs folder to the workspace folder.
So jobs can be submitted like this, type the following command in the admin console:

::
   submit_job ../../job1
   submit_job ../../job2

Participants
------------
Site
____
* :code:`server1`: NVFlare server
* :code:`site_a`: Site_a has a CustomSecurityHandler set up which does not allow the job "FL Demo Job1" to run. Any other named jobs will be able to deploy and run on site_a.
* :code:`site_b`: Site_b does not have the extra security handling codes. It allows any job to be deployed and run.

Jobs
____

* job1: The job is called  :code:`hello-numpy-sag`. site_a will allow this job to run.
* job2: The job is called  :code:`FL Demo Job1`. site_a will block this job to run.


