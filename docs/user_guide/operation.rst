.. _operating_nvflare:

######################################################
Operating NVFLARE - Admin Client, Commands, FLAdminAPI
######################################################

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
    check_status,``check_status server``,"The FL job id, FL server status, and the registered clients with their names and tokens are displayed. If training is running, the round information is also displayed."
    ,``check_status client``,"The name, token, and status of each connected client are displayed."
    ,``check_status client clientname``,"The name, token, and status of the specified client with *clientname* are displayed."
    submit_job,``submit_job job_folder_name``,Submits the job to the server.
    list_jobs,``list_jobs``,Lists the jobs on the server. (Options: [-n name_prefix] [-d] [job_id_prefix])
    abort_job,``abort_job job_id``,Aborts the job of the specified job_id if it is running or dispatched
    clone_job,``clone_job job_id``,Creates a copy of the specified job with a new job_id
    abort,``abort job_id client``,Aborts the job for the specified job_id for all clients. Individual client jobs can be aborted by specifying *clientname*.
    ,``abort job_id server``,Aborts the server job for the specified job_id.
    abort_task,``abort_task job_id clientname``,Aborts the running task for the specified job ID and client.
    download_job,``download_job job_id``,Download folder from the job store containing the job and workspace
    delete_job,``delete_job job_id``,Delete the job from the job store
    cat,``cat server startup/fed_server.json -ns``,Show content of a file (-n: number all output lines; -s: suppress repeated empty output lines)
    ,``cat clientname startup/docker.sh -bT``,Show content of a file (-b: number nonempty output lines; -T: display TAB characters as ^I)
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
    remove_client,``remove_client clientname``,Issue command for server to release client before the 10 minute timeout to allow client to rejoin after manual restart.
    restart,``restart client``,Restarts all of the clients. Individual clients can be restarted by specifying *clientname*.
    ,``restart server``,Restarts the server. Clients will also be restarted. Note that the admin client will need to log in again after the server restarts.
    shutdown,``shutdown client``,Shuts down all of the clients. Individual clients can be shutdown by specifying *clientname*. Please note that this may not be instant but may take time for the command to take effect.
    ,``shutdown server``,Shuts down the active server. Clients must be shut down first before the server is shut down. Note this will not shut down the Overseer or other SPs.
    get_active_sp,``get_active_sp``,Get information on the active SP (service provider or FL server).
    list_sp,``list_sp``,Get data from last heartbeat of the active and available SP endpoint information.
    promote_sp,``promote_sp sp_end_point``,promote a specified SP to become the active SP (promote_sp example1.com:8002:8003)
    shutdown_system,``shutdown_system``,Shut down entire system by setting the system state to shutdown through the overseer


.. note::

   The commands ``promote_sp`` and ``shutdown_system`` both go to the Overseer and have a different mechanism of
   authorization than the other commands sent to the FL server. The Overseer keeps track of a list of privileged users,
   configured to be admin users with the role of "super". Only users owning certificates whose cn is in the privileged
   user list can call these commands.

.. tip::

   Outputs of any command can be redirected into a file by using the greater-than symbol ">", however there must be no
   whitespace before the filename. For example, you may run ``sys_info server >serverinfo.txt``. To only save the
   file output without printing it, use two greater-than symbols ">>" instead: ``sys_info server >>serverinfo.txt``.

.. include:: FLAdminAPI.rst
