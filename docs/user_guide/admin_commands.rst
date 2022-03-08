.. _admin_commands:

##################################
Admin Client, Commands, FLAdminAPI
##################################

The FL system is operated by the packages of type admin configured at provisioning. The admin packages contain key and
certificate files to connect and authenticate with the server, and the administration can be done through an included
command prompt with ``fl_admin.sh`` or programmatically through the FLAdminAPI.

Admin command prompt
====================
After running ``fl_admin.sh``, log in by following the prompt and entering the name of the participant that the admin
package was provisioned for (or for poc mode, "admin" as the name and password).

Typing "help" or "?" will display a list of the commands and a brief description for each. Typing "? " before a command
like "? check_status" or "?ls" will provide additional details for the usage of a command. Provided below is a list of
commands shown as examples of how they may be run with a description.

.. csv-table::
    :header: Command,Example,Description
    :widths: 15, 10, 30

    bye,``bye``,Exit from the client
    help,``help``,Get command help information
    lpwd,``lpwd``,Print local workspace root directory of the admin client
    info,``info``,Show folder setup info (upload and download sources and destinations)
    check_status,``check_status server``,"The FL run number, FL server status, and the registered clients with their names and tokens are displayed. If training is running, the round information is also displayed."
    ,``check_status client``,"The name, token, and status of each connected client are displayed."
    ,``check_status client clientname``,"The name, token, and status of the specified client with *clientname* are displayed."
    upload_app,``upload_app applicationname``,Uploads the application folder to the FL server. Note that *applicationname* is the folder path relative to the "transfer" directory which is at the same level as the "startup" directory containing the script running the admin client.
    set_run_number,``set_run_number 1``,Creates a folder "run_1" on the server at the same level as the "startup" directory to contain all of the applications for deployment.
    deploy_app,``deploy_app applicatonname server``,"Deploys the application specified by *applicationname* to the server. Note that *applicationname* is expected to be an application which has been uploaded to the server already and resides in the *transfer* directory on the server (which is at the same level as the *startup* directory by
    default). *applicationname* can be a relative path if the application is contained in any parent directories, for example apps/segmentation_ct_spleen."
    ,``deploy_app applicationname client``,Deploys the application specified by *applicationname* to each client. This can also be done per client by specifying a specific client name for this command. Please note that the deployed applications are also in their own workspace named after the run number set by set_run_number above.
    start_app,``start_app server``,Starts the server training.
    ,``start_app client``,Starts all of the clients. Individual clients can be started by specifying the client instance name after the start client command.
    abort,``abort client``,Aborts all of the clients. Individual clients can be aborted by specifying *clientname*. Please note that this may not be instant but may take time for the command to take effect.
    ,``abort server``,Aborts the server training
    download_folder,``download_folder foldername``,Download folder from the server's file_download_dir (set to transfer by default)
    cat,``cat server startup/fed_server.json -ns``,Show content of a file (-n: number all output lines; -s: suppress repeated empty output lines)
    ,``cat clientname startup/docker.sh -bT``,Show content of a file (-b: number nonempty output lines; -T: display TAB characters as ^I)
    env,``env server``,Show environment variables
    ,``env clientname``,Show environment variables
    grep,``grep server "info" -i log.txt``,Search for a pattern in a file (-n: print line number; -i: ignore case)
    head,``head clientname log.txt``,Print the first 10 lines of a file
    ,``head server log.txt -n 15``,Print the first 15 lines of a file (-n: print the first N lines instead of the first 10)
    tail,``tail clientname log.txt``,Print the last 10 lines of a file
    ,``tail server log.txt -n 15``,Print the last 15 lines of a file (-n: output the last N lines instead of the last 10)
    ls,``ls server -alt``,List files in workspace root directory (-a: all; -l: use a long listing format; -t: sort by modification time)
    ,``ls clientname -SR``,List files in workspace root directory (-S: sort by file size; -R: list subdirectories recursively)
    pwd,``pwd server``,Print the name of workspace root directory
    ,``pwd clientname``,Print the name of workspace root directory
    sys_info,``sys_info server``,Get system information
    ,``sys_info client *clientname*``,Get system information. Individual clients can be shutdown by specifying *clientname*.
    restart,``restart client``,Restarts all of the clients. Individual clients can be restarted by specifying *clientname*.
    ,``restart server``,Restarts the server. Clients will also be restarted. Note that the admin client will need to log in again after the server restarts.
    shutdown,``shutdown client``,Shuts down all of the clients. Individual clients can be shutdown by specifying *clientname*. Please note that this may not be instant but may take time for the command to take effect.
    ,``shutdown server``,Shuts down the server. Clients must be shut down first before the server is shut down.


.. tip::

   Outputs of any command can be redirected into a file by using the greater-than symbol ">", however there must be no
   whitespace before the filename. For example, you may run ``sys_info server >serverinfo.txt``. To only save the
   file output without printing it, use two greater-than symbols ">>" instead: ``sys_info server >>serverinfo.txt``.

.. include:: FLAdminAPI.rst
