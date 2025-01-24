*****************************************
Dashboard Command
*****************************************

Introduction to the Dashboard Command
=====================================

The Dashboard command allows users to start the :ref:`dashboard_api` to provide a simple way to collect information
of clients and users from different organizations and generate startup kits for users to download.

Syntax and Usage
=================

Running ``nvflare dashboard -h`` shows all available options.

.. code-block:: shell

    (nvflare_venv) ~/workspace/repos/flare$ nvflare dashboard -h
    usage: nvflare dashboard [-h] [--start] [--stop] [-p PORT] [-f FOLDER] [-i DASHBOARD_IMAGE] [--passphrase PASSPHRASE] [-e ENV]

    options:
    -h, --help            show this help message and exit
    --cloud CLOUD         launch dashboard on cloud service provider (ex: --cloud azure or --cloud aws)
    --start               start dashboard
    --stop                stop dashboard
    -p PORT, --port PORT  port to listen
    -f FOLDER, --folder FOLDER
                            folder containing necessary info (default: current working directory)
    --passphrase PASSPHRASE
                            Passphrase to encrypt/decrypt root CA private key. !!! Do not share it with others. !!!
    -e ENV, --env ENV     additional environment variables: var1=value1
    --cred CRED           set credential directly in the form of USER_EMAIL:PASSWORD
    -i IMAGE, --image IMAGE
                            set the container image name
    --local               start dashboard locally without docker image


To start Dashboard, run ``nvflare dashboard --start``.

The Dashboard Docker will detect if the database is initialized.  If not, it will ask for the project_admin email address and will generate a random password:

.. code-block::

    Please provide project admin email address.  This person will be the super user of the dashboard and this project.
    project_admin@admin_organization.com
    generating random password
    Project admin credential is project_admin@admin_organization.com and the password is EXAMPLE1

Please log in with this credential to finish setting up the project in Dashboard once the system is up and running.
The project_admin can change his/her password in the Dashboard system after logging in.

Note that for the first time, it may take a while to download the nvflare image as you see the prompt:

.. code-block::

    Pulling nvflare/nvflare, may take some time to finish.

After pulling the image, you should see output similar to the following:

.. code-block::

    Launching nvflare/nvflare
    Dashboard will listen to port 443
    /path_to_folder_for_db on host mounted to /var/tmp/nvflare/dashboard in container
    No additional environment variables set to the launched container.
    Dashboard container started
    Container name nvflare-dashboard
    id is 3108eb7be20b92ab3ec3dd7bfa86c2eb83bd441b4da0865d2ebb10bd60612345

We suggest you to set the passphrase to protect the private key of the root CA by using the ``--passphrase`` option.  Once it's set, you have to provide the same passphrase everytime you
restart Dashboard for the same project.

If you would like to start a new project, please remove the db.sqlite file in current working directory (or the directory set with the ``--folder`` option).  Dashboard will start
from scratch and you can provide a project admin email address and get a new password for the project_admin.

The Dashboard will also check the cert folder inside current the working directory (or directory specified by the --folder option) to load web.crt and web.key.
If those files exist, Dashboard will load them and run as an HTTPS server.  If Dashboard does not find both of them, it runs as HTTP server.  In both cases, the service
listens to port 443, unless the ``--port`` option is used to specify a different port. Dashboard will run on ``0.0.0.0``, so by default it should be accessible on the same machine from
``localhost:443``. To make it available to users outside the network, port forwarding and other configurations may be needed to securely direct traffic to the machine running Dashboard.

.. note::

    Running Dashboard requires Docker. You have to ensure your system can pull and run Docker images. The initial docker pull may take some time depending on your network connection.

To stop the running Dashboard, run ``nvflare dashboard --stop``.
