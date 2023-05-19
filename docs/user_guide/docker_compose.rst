.. _docker_compose:

######################################################
Launching NVIDIA FLARE with docker compose
######################################################

For users who would like to get NVIDIA FLARE up and running as easy as possible,
such as first-time NVIDIA FLARE users or people who need to demonstrate it upon request,
they can use this docker compose feature.  All they need is a working docker 
environment.

The provisioning tool of NVIDIA FLARE now includes a new
builder ``DockerBuilder`` that can create ``compose.yaml`` and other information.  
After provisioing, users can enter the result folder, normally in 
workspace/example_project/prod_NN, and type ``docker compose build`` 
and ``docker compose up`` to start overseer, servers and clients 
in the docker compose manner.


Provisioning stage
==================
First check if your project.yml file contains the following section.

.. code-block:: yaml

  - path: nvflare.lighter.impl.docker.DockerBuilder
    args:
      base_image: python:3.8
      requirements_file: docker_compose_requirements.txt


This builder will generate the necessary information during provisioning time.

The ``base_image`` argument is the base docker image name that will be used to create
the runtime docker image for NVIDIA FLARE in docker compose setting.

The ``requirements_file`` can contain additional python packages that will be installed
after nvflare package is installed in the runtime docker image.  If you don't need to install
any additional python package, you can provide an empty file.


Post-provisioning stage
=======================

Running provision command as usual, either in the new format ``nvflare provision`` or just ``provision``.

After the command, there should a folder with structure similar to the following:

.. code-block:: shell

    $ tree -L 1
    .
    ├── admin@nvidia.com
    ├── compose.yaml
    ├── nvflare_compose
    ├── nvflare_hc
    ├── overseer
    ├── server1
    ├── server2
    ├── site-1
    └── site-2

    8 directories, 1 file


The ``compose.yaml`` is the key file for docker compose command and the folder ``nvflare_compose`` 
is the compose context folder for generating runtime docker image during ``docker compose build`` stage.

The content inside ``nvflare_compose`` consists of two files only, ``Dockerfile`` and ``requirements.txt``.
You can modify them if necessary.  For example, if you need to install additional binary packages with ``apt-get install``,
you can add them in the Dockerfile.

The ``requirements.txt`` is a copy of the requirements_file you provided in the project.yml file.


Running docker compose
=======================

Inside the prod_NN folder, if this is the very first time you start the docker compose for NVIDIA FLARE, please
run ``docker compose build`` to build the runtime docker image.  If nothing is changed in Dockerfile and requirements.txt,
you don't have to run that command again.

.. code-block:: shell

    $ docker compose build
    [+] Building 0.1s (10/10) FINISHED                                                                                                                                                                                                       
    => [internal] load build definition from Dockerfile                                                                                                                                                                                0.0s
    => => transferring dockerfile: 177B                                                                                                                                                                                                0.0s
    => [internal] load .dockerignore                                                                                                                                                                                                   0.0s
    => => transferring context: 2B                                                                                                                                                                                                     0.0s
    => [internal] load metadata for docker.io/library/python:3.8                                                                                                                                                                       0.0s
    => [1/5] FROM docker.io/library/python:3.8                                                                                                                                                                                         0.0s
    => [internal] load build context                                                                                                                                                                                                   0.0s
    => => transferring context: 37B                                                                                                                                                                                                    0.0s
    => CACHED [2/5] RUN pip install -U pip                                                                                                                                                                                             0.0s
    => CACHED [3/5] RUN pip install nvflare                                                                                                                                                                                            0.0s
    => CACHED [4/5] COPY requirements.txt requirements.txt                                                                                                                                                                             0.0s
    => CACHED [5/5] RUN pip install -r requirements.txt                                                                                                                                                                                0.0s
    => exporting to image                                                                                                                                                                                                              0.0s
    => => exporting layers                                                                                                                                                                                                             0.0s
    => => writing image sha256:53a1463bd170b8bc213899037bbe4403f2d6f0d553cdd470805855f3968d19d4                                                                                                                                        0.0s
    => => naming to docker.io/library/nvflare-service                                                                                                                                                                                  0.0s

After the runtime docker image is ready, you can run ``docker compose up`` to get one overseer, two servers and two sites
running together.  The ports for overseer and servers are also opened.  The overseer/severs/clients folders in current 
prod_NN folder are mounted to different running docker instances.  An internal folder will be mounted by servers to store
shared snapshot information.

.. code-block:: shell

    $ docker compose up
    [+] Running 5/0
    ⠿ Container prod_02-site-1-1    Recreated                                                                                                                                                                                          0.1s
    ⠿ Container prod_02-overseer-1  Recreated                                                                                                                                                                                          0.1s
    ⠿ Container prod_02-server1-1   Recreated                                                                                                                                                                                          0.1s
    ⠿ Container prod_02-server2-1   Recreated                                                                                                                                                                                          0.1s
    ⠿ Container prod_02-site-2-1    Recreated                                                                                                                                                                                          0.1s
    Attaching to prod_02-overseer-1, prod_02-server1-1, prod_02-server2-1, prod_02-site-1-1, prod_02-site-2-1
    prod_02-overseer-1  | [2022-09-23 16:00:58 +0000] [9] [INFO] Starting gunicorn 20.1.0
    prod_02-overseer-1  | [2022-09-23 16:00:58 +0000] [9] [INFO] Listening at: https://0.0.0.0:8443 (9)
    prod_02-overseer-1  | [2022-09-23 16:00:58 +0000] [9] [INFO] Using worker: nvflare.ha.overseer.worker.ClientAuthWorker
    prod_02-overseer-1  | [2022-09-23 16:00:58 +0000] [12] [INFO] Booting worker with pid: 12
    prod_02-server2-1   | 2022-09-23 16:00:59,103 - FederatedServer - INFO - starting secure server at server2:8102
    prod_02-server2-1   | deployed FL server trainer.
    prod_02-server2-1   | 2022-09-23 16:00:59,118 - nvflare.fuel.hci.server.hci - INFO - Starting Admin Server server2 on Port 8103
    prod_02-server2-1   | 2022-09-23 16:00:59,119 - root - INFO - Server started
    prod_02-server2-1   | 2022-09-23 16:00:59,121 - FederatedServer - INFO - Got the primary sp: server2 fl_port: 8102 SSID: 9ba168f0-6cf5-446b-bfd5-a1243dd195f8. Turning to hot.
    prod_02-server1-1   | 2022-09-23 16:00:59,332 - FederatedServer - INFO - starting secure server at server1:8002
    prod_02-server1-1   | deployed FL server trainer.
    prod_02-server1-1   | 2022-09-23 16:00:59,346 - nvflare.fuel.hci.server.hci - INFO - Starting Admin Server server1 on Port 8003
    prod_02-server1-1   | 2022-09-23 16:00:59,346 - root - INFO - Server started
    prod_02-site-2-1    | Waiting for SP....
    prod_02-site-2-1    | 2022-09-23 16:00:59,399 - FederatedClient - INFO - Got the new primary SP: server2:8102
    prod_02-site-1-1    | Waiting for SP....
    prod_02-site-1-1    | 2022-09-23 16:00:59,450 - FederatedClient - INFO - Got the new primary SP: server2:8102
    prod_02-server2-1   | 2022-09-23 16:01:00,393 - ClientManager - INFO - Client: New client site-2@172.18.0.2 joined. Sent token: 3da72f67-3443-47ac-b059-76b0b314dd08.  Total clients: 1
    prod_02-site-2-1    | 2022-09-23 16:01:00,394 - FederatedClient - INFO - Successfully registered client:site-2 for project example_project. Token:3da72f67-3443-47ac-b059-76b0b314dd08 SSID:9ba168f0-6cf5-446b-bfd5-a1243dd195f8
    prod_02-server2-1   | 2022-09-23 16:01:00,439 - ClientManager - INFO - Client: New client site-1@172.18.0.3 joined. Sent token: 5e0b1012-77e6-41a3-8af0-9fa86df8ef2e.  Total clients: 2
    prod_02-site-1-1    | 2022-09-23 16:01:00,440 - FederatedClient - INFO - Successfully registered client:site-1 for project example_project. Token:5e0b1012-77e6-41a3-8af0-9fa86df8ef2e SSID:9ba168f0-6cf5-446b-bfd5-a1243dd195f8

Login with admin console
========================
You can use admin console to login to this newly created NVIDIA FLARE system after your machine can resolve the IP
addresses of overseer and servers.  For example, if you are running the docker compose at machine ``desktop1`` with ip 192.168.1.101 and 
would like to run your admin console at machine ``desktop2``, you will need to edit the /etc/hosts file on desktop2 to include this line:

.. code-block::

    192.168.1.101 overseer server1 server2

After this update, the admin console can find overseer, server1 and server2.  If in your project.yml file, 
you name them differently, for example myoverseer for the overseer, please change that line to

.. code-block::

    192.168.1.101 myoverseer server1 server2


Login with admin console will be as usual.  Just run fl_admin.sh in the startup folder of admin console startup.

.. code-block:: shell
    
    $ ./admin@nvidia.com/startup/fl_admin.sh 
    User Name: admin@nvidia.com
    Trying to obtain server address
    Obtained server address: server1:8003
    Trying to login, please wait ...
    Logged into server at server1:8003
    Type ? to list commands; type "? cmdName" to show usage of a command.
    > check_status server
    Engine status: stopped
    ---------------------
    | JOB_ID | APP NAME |
    ---------------------
    ---------------------
    Registered clients: 2 
    ----------------------------------------------------------------------------
    | CLIENT | TOKEN                                | LAST CONNECT TIME        |
    ----------------------------------------------------------------------------
    | site-2 | 7cfe5dce-00a5-4ffb-a5ad-d31dc050c5dd | Fri Sep 23 16:15:00 2022 |
    | site-1 | 5435ccb6-9240-42b1-a48b-6290cc71d8d0 | Fri Sep 23 16:15:00 2022 |
    ----------------------------------------------------------------------------
    Done [9729 usecs] 2022-09-23 09:15:12.137237

Ending docker compose
=====================

You can press ``CTRL-C`` to stop the docker compose.