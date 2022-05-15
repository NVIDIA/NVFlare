FLAdminAPI
==========

:class:`FLAdminAPI<nvflare.fuel.hci.client.fl_admin_api.FLAdminAPI>` is a wrapper for admin commands that can be issued
by an admin client to the FL server. You can use a provisioned admin client's certs and keys to initialize an instance
of FLAdminAPI to programmatically submit commands to the FL server.

Initialization
--------------

Initialize the API with actual values for the FL setup: host, port, paths to files and directories.

A provisioned admin package should have ca_cert, client_cert, and client_key in the startup folder, and transfer can be
created at the same level as startup.

Log in with the admin name that corresponds to the provisioned package.

.. code:: python

    from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
    api = FLAdminAPI(
        host=host,
        port=port,
        ca_cert=ca_cert,
        client_cert=client_cert,
        client_key=client_key,
        upload_dir=upload_dir,
        download_dir=download_dir
    )
    api.login(username="admin@nvidia.com")

After using FLAdminAPI, the ``logout()`` function can be called to log out. Both ``login()`` and ``logout()`` are
inherited from AdminAPI.

.. code:: python

    api.logout()

Usage
-----
Simplest sequence to upload, deploy, and start training with the "hello-pt" example app:

.. code:: python

    api.upload_app("hello-pt")
    api.set_run_number(1)
    api.deploy_app("hello-pt", "all")
    api.start_app("all")

Contents of the returned FLAdminAPIResponse can be accessed:

.. code:: python

    reply = api.upload_app("hello-pt")
    response_text = reply["details"]["message"]

Arguments and targets
---------------------
The arguments required for each FLAdminAPI call are specified in the functions section below.

The ``app`` is the name of the directory to upload and deploy containing configurations and custom code.

The ``target`` when needed is where the action should take place. An argument of ``target`` as a string can be a
singular target of "server" or a specific client name. Where ``target_type`` is required and ``targets`` can
be an optional list, the call can be submitted to multiple targets:

- If ``target_type`` is "server", the command target is just the server and ``targets`` is ignored
- If ``target_type`` is "client" and ``targets`` is empty, the command target is all clients
- If ``target_type`` is "client" and ``targets`` is a list of strings, each a client name, the command target is all clients in the list ``targets``
- If ``target_type`` is "all" and the command supports it, the command target is the server and all clients

Return Structure
----------------
FLAdminAPI calls return an FLAdminAPIResponse dictionary object of key value pairs consisting of a status of type APIStatus, a
dictionary with details, and in some cases a raw response from the underlying call to AdminAPI (mainly useful for
debugging). The possible statuses and contents of the details returned are outlined in each of the function calls below.

Note that FLAdminAPI uses the underlying AdminAPI's ``do_command()`` function to submit commands to the server, and you
can also use this function directly for functions that are not wrapped in an FLAdminAPI function. Returns from AdminAPI
are included in the FLAdminAPI reply under the "raw" key for some calls and error conditions.

Basic Functions
---------------

``upload_app(app)``
^^^^^^^^^^^^^^^^^^^
Uploads app to upload directory of FL server.

+-------+--------------+----------+------------+-------------------------------------------+
|       | Argument     | Type     | Required   | Description                               |
+=======+==============+==========+============+===========================================+
| 1     | app          | str      | yes        | name of the folder in upload_dir to upload|
+-------+--------------+----------+------------+-------------------------------------------+

Returns:

* Status SUCCESS - successful upload of app to server
* Status ERROR_RUNTIME - app directory may not be located, or other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``set_run_number(run_number)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sets a run number in order to keep track of the current experiment to deploy and start an app.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | run_number | integer   | yes        | run number to set for the current experiment|
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful creation of new run number
* Status PARTIAL_SUCCESS - run number already exists but is now set
* Status ERROR_SYNTAX - run number must be an integer
* Status ERROR_RUNTIME - run number may not be changed during training, or other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``deploy_app(app, target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Deploys specified app to run number specific instance on specified targets. Target of "all" deploys to server then
clients. Run number must be set and app must be uploaded to server already.

+-------+-------------+-----------+------------+--------------------------------------------------------------------+
|       | Argument    | Type      | Required   | Description                                                        |
+=======+=============+===========+============+====================================================================+
| 1     | app         | str       | yes        | name of app to deploy                                              |
+-------+-------------+-----------+------------+--------------------------------------------------------------------+
| 2     | target_type | str       | yes        | can be server, client, or all                                      |
+-------+-------------+-----------+------------+--------------------------------------------------------------------+
| 3     | targets     | list      | no         | list of client names each type str, if target_type is client       |
+-------+-------------+-----------+------------+--------------------------------------------------------------------+

Returns:

* Status SUCCESS - successful deployment of app (partial success of deployment to some but not all targets is included)
* Status ERROR_SYNTAX - app and target must be type string
* Status ERROR_INVALID_CLIENT - server replies that the specified target is invalid
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``start_app(target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Starts the currently deployed app at the specified target.

+-------+--------------+----------+------------+--------------------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                                  |
+=======+==============+==========+============+==============================================================+
| 1     | target_type  | str      | yes        | can be server, client, or all                                |
+-------+--------------+----------+------------+--------------------------------------------------------------+
| 2     | targets      | list     | no         | list of client names each type str, if target_type is client |
+-------+--------------+----------+------------+--------------------------------------------------------------+

Returns:

* Status SUCCESS - successful start of app (partial success of some but not all targets is included if server is in targets)
* Status PARTIAL_SUCCESS - at least one target is already started
* Status ERROR_SYNTAX - target must be type string
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``abort(target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Stops the app at the specified targets. If the target is server and the clients are not stopped, the server will try to
abort clients first, and the clients may enter cross validation from a status of in training. More functions involving
abort in the advanced functions section below address some of this.

+-------+--------------+----------+------------+--------------------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                                  |
+=======+==============+==========+============+==============================================================+
| 1     | target_type  | str      | yes        | can be server or client                                      |
+-------+--------------+----------+------------+--------------------------------------------------------------+
| 2     | targets      | list     | no         | list of client names each type str, if target_type is client |
+-------+--------------+----------+------------+--------------------------------------------------------------+

Returns:

* Status SUCCESS - successful abort of app (partial success of some but not all targets is included; abort server may try to abort clients first, and clients may enter cross validation from training and not stop immediately)
* Status PARTIAL_SUCCESS - training not started
* Status ERROR_SYNTAX - target must be type string
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``shutdown(target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Shuts down the specified targets to stop running FL. The server must be shut down after the clients.

+-------+--------------+----------+------------+--------------------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                                  |
+=======+==============+==========+============+==============================================================+
| 1     | target_type  | str      | yes        | can be server, client, or all                                |
+-------+--------------+----------+------------+--------------------------------------------------------------+
| 2     | targets      | list     | no         | list of client names each type str, if target_type is client |
+-------+--------------+----------+------------+--------------------------------------------------------------+

Returns:

* Status SUCCESS - successful shutdown of app (partial success of some but not all targets is included)
* Status ERROR_SYNTAX - target must be type string
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

Convenience Functions
---------------------

``check_status(target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Checks and returns the FL status. If target_type is server, the call does not wait for the server to retrieve
information on the clients but returns the last information the server had at the time this call is made.

If target_type is client, specific clients can be specified in targets, and this call generally takes longer than the
function to just check the FL server status because this one waits for communication from the server to client then
back.

+-------+--------------+----------+------------+--------------------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                                  |
+=======+==============+==========+============+==============================================================+
| 1     | target_type  | str      | yes        | can be server or client                                      |
+-------+--------------+----------+------------+--------------------------------------------------------------+
| 2     | targets      | list     | no         | list of client names each type str, if target_type is client |
+-------+--------------+----------+------------+--------------------------------------------------------------+

Returns:

* Status SUCCESS - successful status check (the structure of the details returned depends on what is relevant for the status)\
* Status ERROR_RUNTIME - other runtime error

With following details structure for target_type server and server statuses training not started and training stopped:

+--------------------+----------+-------------------------------------------------------------------+
| Key                | Type     | Description                                                       |
+====================+==========+===================================================================+
| run_number         | str      | current set run number, or has not been set                       |
+--------------------+----------+-------------------------------------------------------------------+
| server_status      | str      | status of the server                                              |
+--------------------+----------+-------------------------------------------------------------------+
| registered_clients | int      | number of clients that connected and registered on the server     |
+--------------------+----------+-------------------------------------------------------------------+
| status_table       | list     | table of server's last client statuses                            |
+--------------------+----------+-------------------------------------------------------------------+

With target_type server and server status training started, the details structure is:

+--------------------+----------+-------------------------------------------------------------------+
| Key                | Type     | Description                                                       |
+====================+==========+===================================================================+
| run_number         | str      | current set run number                                            |
+--------------------+----------+-------------------------------------------------------------------+
| server_status      | str      | status of the server                                              |
+--------------------+----------+-------------------------------------------------------------------+
| start_round        | str      | the round for the server to start FL at                           |
+--------------------+----------+-------------------------------------------------------------------+
| max_round          | str      | the round after which the server will stop FL                     |
+--------------------+----------+-------------------------------------------------------------------+
| min_num_clients    | int      | minimum number of clients for the server to start aggregation     |
+--------------------+----------+-------------------------------------------------------------------+
| max_num_clients    | int      | maximum number of clients that can connect to the server          |
+--------------------+----------+-------------------------------------------------------------------+
| registered_clients | int      | number of clients that connected and registered on the server     |
+--------------------+----------+-------------------------------------------------------------------+
| submitted_models   | int      | number of clients that submitted models for the current round     |
+--------------------+----------+-------------------------------------------------------------------+
| status_table       | list     | table of server's last client statuses                            |
+--------------------+----------+-------------------------------------------------------------------+

With target_type client, the details structure is:

+--------------------+----------+-------------------------------------------------------------------+
| Key                | Type     | Description                                                       |
+====================+==========+===================================================================+
| fl_run_number      | str      | current set run number, or has not been set                       |
+--------------------+----------+-------------------------------------------------------------------+
| server_status      | str      | status of the server                                              |
+--------------------+----------+-------------------------------------------------------------------+
| registered_clients | int      | number of clients that connected and registered on the server     |
+--------------------+----------+-------------------------------------------------------------------+
| status_table       | list     | table of server's last client statuses                            |
+--------------------+----------+-------------------------------------------------------------------+

``get_validation_results(target)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Gets the validation results for the current run number from the server. This is only relevant if cross validation had
been configured in the app and has been allowed to run, otherwise the results will be empty. If the target is not
included, it will default to all and this will return the full validation results on the server. The target can also be
two client names to get just the result of the first client's model on the second client's data.

+-------+--------------+----------+------------+------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                    |
+=======+==============+==========+============+================================================+
| 1     | target       | str      | no         | can be all, or <client-name1> <client-name2>   |
+-------+--------------+----------+------------+------------------------------------------------+

Returns:

* Status SUCCESS - successful return of validation results
* Status ERROR_SYNTAX - target must be type string if included
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``restart(target_type, targets)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Restarts the specified targets. Restarting the server will restart all of the clients too. After restarting the server,
you will need to log in again in order to issue commands.

+-------+--------------+----------+------------+--------------------------------------------------------------+
|       | Argument     | Type     | Required   | Description                                                  |
+=======+==============+==========+============+==============================================================+
| 1     | target_type  | str      | yes        | can be server, client, or all                                |
+-------+--------------+----------+------------+--------------------------------------------------------------+
| 2     | targets      | list     | no         | list of client names each type str, if target_type is client |
+-------+--------------+----------+------------+--------------------------------------------------------------+

Returns:

* Status SUCCESS - successful submission of restart command
* Status ERROR_SYNTAX - target must be type string
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``remove_client(target)``
^^^^^^^^^^^^^^^^^^^^^^^^^
Removes the specified targets.
Note that if a client is removed, you will not be able to issue admin commands through the server to that client
until the client is restarted (this includes being able to issue the restart command through the API).

+-------+--------------+----------+------------+---------------------------------------------+
|       | Argument     | Type     | Required   | Description                                 |
+=======+==============+==========+============+=============================================+
| 1     | targets      | list     | yes        | list of client names each type str          |
+-------+--------------+----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful submission of restart command
* Status ERROR_SYNTAX - target must be type string
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``delete_run_number(run_number)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Deletes the run folder corresponding to the run number on the server and all connected clients. This is not reversible.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | run_number | int       | yes        | run number for the run to delete            |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful creation of new run number
* Status ERROR_SYNTAX - run number must be an integer
* Status ERROR_RUNTIME - run number may not be changed during training, or other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``set_timeout(timeout)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sets the timeout for admin commands in seconds. This timeout is the maximum amount of time the server will wait for
replies from clients. If the timeout is too short, the server may not receive a response because clients may not have a
chance to reply.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | timeout    | float     | yes        | timeout to set in seconds                   |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful setting of timeout
* Status ERROR_SYNTAX - timeout must be type float
* Status ERROR_RUNTIME - other runtime error

All with following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``ls_target(target, options, path)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Retrieves the contents of the path (relative to the working directory of the specified target). If no path is specified,
the contents of the working directory are returned. The target can be "server" or a specific client name for example
"org2". The allowed options are: "-a" for all, "-l" to use a long listing format, "-t" to sort by modification time
newest first, "-S" to sort by file size largest first, "-R" to list subdirectories recursively, "-u" with -l to show
access time otherwise sort by access time.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | target     | str       | yes        | can be server or <client-name>              |
+-------+------------+-----------+------------+---------------------------------------------+
| 2     | options    | str       | no         | see allowed options above                   |
+-------+------------+-----------+------------+---------------------------------------------+
| 3     | path       | str       | no         | optional path to a directory                |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful return from server
* Status ERROR_SYNTAX - target must be included and type str
* Status ERROR_INVALID_CLIENT - server replies that the specified target is invalid
* Status ERROR_RUNTIME - other runtime error

With following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``cat_target(target, options, file)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sends the shell command to get the contents of the target's specified file. The target can be "server" or a specific
client name for example "org2". The file is required and should contain the relative path to the file from the working
directory of the target. The allowed options are "-n" to number all output lines, "-b" to number nonempty output lines,
"-s" to suppress repeated empty output lines, and "-T" to display TAB characters as ^I.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | target     | str       | yes        | can be server or <client-name>              |
+-------+------------+-----------+------------+---------------------------------------------+
| 2     | options    | str       | no         | see allowed options above                   |
+-------+------------+-----------+------------+---------------------------------------------+
| 3     | file       | str       | yes        | path to the file to return the contents of  |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful return from server
* Status ERROR_SYNTAX - target and file must be included and type str
* Status ERROR_INVALID_CLIENT - server replies that the specified target is invalid
* Status ERROR_RUNTIME - other runtime error

With following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``tail_target_log(target, options)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Returns end of target's log allowing for options that the admin client allows: "-n" can be used to specify the
number of lines for example "-n 100", or "-c" can specify the number of bytes.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | target     | str       | yes        | can be server or <client-name>              |
+-------+------------+-----------+------------+---------------------------------------------+
| 2     | options    | str       | no         | "-n" for number of lines and "-c" for bytes |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful log return
* Status ERROR_RUNTIME - other runtime error

With following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``grep_target(target, options, pattern, file)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Sends the grep shell command to search the contents of the target's specified file. The target can be "server" or a
specific client name for example "org2". The file is required and should contain the relative path to the file from the
working directory of the target. The pattern is also required. The allowed options are "-n" to print line number with
output lines, "-i" to ignore case distinctions, and "-b" to print the byte offset with output lines.

+-------+------------+-----------+------------+---------------------------------------------+
|       | Argument   | Type      | Required   | Description                                 |
+=======+============+===========+============+=============================================+
| 1     | target     | str       | yes        | can be server or <client-name>              |
+-------+------------+-----------+------------+---------------------------------------------+
| 2     | options    | str       | no         | see allowed options above                   |
+-------+------------+-----------+------------+---------------------------------------------+
| 3     | pattern    | str       | yes        | the pattern to search for                   |
+-------+------------+-----------+------------+---------------------------------------------+
| 4     | file       | str       | yes        | path to the file to return the contents of  |
+-------+------------+-----------+------------+---------------------------------------------+

Returns:

* Status SUCCESS - successful return from server
* Status ERROR_SYNTAX - target, pattern, and file must be included and type str
* Status ERROR_RUNTIME - other runtime error

With following details structure:

+--------------+----------+-------------------------------------------------------------------+
| Key          | Type     | Description                                                       |
+==============+==========+===================================================================+
| message      | str      | reply string from server                                          |
+--------------+----------+-------------------------------------------------------------------+

``wait_until_(interval, timeout, callback, fail_attempts, **kwargs)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The functions ``wait_until_server_status()``, ``wait_until_client_status()``, and ``wait_until_server_stats()`` are
included with the FLAdminAPI in NVIDIA FLARE as examples of useful functions that can be built with other calls in a
loop with logic. These examples wait until the provided callback returns True, with the option to specify a timeout and
interval to check the status or stats. There is a default callback to evaluate the reply in the included functions, and
additional kwargs passed in will be available to the callback. Custom callbacks can be provided to add logic to handle
checking for other conditions. For these example functions, a timeout should be set in case there are any error
conditions that result in the system being stuck in a state where the callback never returns True.

You can use the source code of these function as inspiration to create your own functions or logic that makes use of
other FLAdminAPI calls.

+-------+---------------+-----------+------------+---------------------------------------------------------------------+
|       | Argument      | Type      | Required   | Description                                                         |
+=======+===============+===========+============+=====================================================================+
| 1     | interval      | int       | no         | interval in seconds between checks of status to provide to callback |
+-------+---------------+-----------+------------+---------------------------------------------------------------------+
| 2     | timeout       | int       | no         | time in seconds to run before returning with timeout message        |
+-------+---------------+-----------+------------+---------------------------------------------------------------------+
| 3     | callback      | Callable  | no         | callback to determine condition to end this call and return         |
+-------+---------------+-----------+------------+---------------------------------------------------------------------+
| 4     | fail_attempts | int       | no         | number of consecutive failed attempts of getting the status         |
+-------+---------------+-----------+------------+---------------------------------------------------------------------+

Questions
---------

#. Why do I get an error of "Command ___ not found in server or client cmds" even though I did not try any unspecified
   command?

   The underlying AdminAPI may have not have successfully logged in and obtained a list of available commands to register
   from the server. Please make sure that the server is accessible and the login is working.

#. Why does the AdminAPI return status APIStatus.SUCCESS even though an error occurred after issuing the command?

   If you send a raw command to the underlying AdminAPI with ``do_command()``, AdminAPI returns APIStatus.SUCCESS if the
   command was successfully sent to the server and a reply obtained. FLAdminAPI's calls make sense of the underlying
   server reply and returns a suitable status based on the reply.

#. After a while with the same command, why do I get a SUCCESS from FLAdminAPI but the raw reply contains an error of
   "not authenticated - no user"?

   The server has a timeout after which ``login()`` must be called again in order for the underlying AdminAPI to be
   authenticated.
