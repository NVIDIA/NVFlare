Example for Federated Policies
==============================


Overview
--------

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Show secure admin client and authentication
3. Demonstrate local authorization policy
4. Demonstrate local privacy policy

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
    cd NVFlare/examples/advanced/federated-policies
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

In secure mode, NVFlare creates one startup kit for each user. There are 5 users in
this example so there are 5 folders for admin login under :code:`workspace/fed_policy/prod_00` folder.

To login as an user, the appropriate folder must be selected.

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
* :code:`site_a`: Client owned by a.org with a customized authorization policy, which only allows
users from the same org to submit job.
* :code:`site_b`: Client owned by b.org with a customized privacy policy. The policy defines
two scopes :code:`public` and :code:`private`. A custom filter is applied to :code:`private`.

Users
_____
* :code:`super@a.org`: Super user with role :code:`project_admin` who can do everything
* :code:`admin@a.org`: Admin for a.org with role :code:`org_admin`
* :code:`trainer@a.org`: Lead trainer for a.org with role :code:`lead`
* :code:`trainer@b.org`: Lead trainer for b.org with role :code:`lead`
* :code:`user@b.org`: Regular user for b.org with role :code:`member`

Jobs
____
All the jobs run the same app (numpy-sag) but have different scopes defined in :code:`meta.json`.

* job1: Scope is :code:`public`. No filters.
* job2: Scope is :code:`test`. Test filters are applied to data and result.
* job3: Scope is :code:`private`. PercentilePrivacy filter is applied to result.
* job4: It has no scope defined.
* job5: It defines an non-existent scope :code:`foo`


Test Cases
----------

Authorization
_____________
We will demo some authorization behaviors.

Since authorization decision is determined using each site's authorization.json and each admin user's role,
we just use :code:`job1` in all the following tests.

.. list-table:: Authorization Use Cases
    :widths: 14 20 50
    :header-rows: 1

    * - User
      - Command
      - Expected behavior
    * - trainer@a.org
      - submit_job ../../job1
      - Job deployed and started on all sites
    * - trainer@a.org
      - clone_job [the job ID that we previous submitted]
      - Job deployed and started on all sites
    * - trainer@b.org
      - clone_job [the job ID that we previous submitted]
      - Rejected because submitter is in a different org
    * - admin@a.org
      - submit_job ../../job1
      - Rejected because role "org_admin" is not allowed to submit jobs
    * - trainer@b.org
      - submit_job ../../job1
      - site_a rejected the job because the submitter is in a different org, while site_b accepted the job
        so the job will still run since in meta.json we specify min_clients as 1

Privacy
_______
site_a has no privacy policy defined.
So we will test the following cases on site_b.

In each job's meta.json we specified their "scope" and in site's privacy.json file each site will define its own
privacy filters to apply for that scope.

Note that default jobs are treated in "public" scope.

Let's just use user trainer@b.org for the following tests.

.. list-table:: Privacy Policy Use Cases
    :widths: 10 50
    :header-rows: 1

    * - Job
      - Expected behavior
    * - job1
      - Job deployed with no filters
    * - job2
      - Job deployed with TestFilter applied
    * - job3
      - Job deployed with PercentilePrivacy filter applied to the result
    * - job4
      - Job deployed using default scope :code:`public`
    * - job5
      - Job rejected by site_b because :code:`foo` doesn't exist

Shutting down NVFlare
_____________________
All NVFlare server and clients can be stopped by using this script,
::
   ./stop.sh
