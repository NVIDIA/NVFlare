FLAdminAPI
==========

:class:`FLAdminAPI<nvflare.fuel.hci.client.fl_admin_api.FLAdminAPI>` is a wrapper for admin commands that can be issued
by an admin client to the FL server. You can use a provisioned admin client's certs and keys to initialize an instance
of FLAdminAPI to programmatically submit commands to the FL server.

Initialization
--------------
It is recommended to use the :class:`FLAdminAPIRunner<nvflare.fuel.hci.client.fl_admin_api_runner.FLAdminAPIRunner>` to
initialize the API, or use it as a guide to write your own code to use the FLAdminAPI.

Compared to before NVIDIA FLARE 2.1.0, the FLAdminAPI now requires an overseer_agent to be provided, and this is automatically
created by the :class:`FLAdminAPIRunner<nvflare.fuel.hci.client.fl_admin_api_runner.FLAdminAPIRunner>` with the
information in ``fed_admin.json`` in the provided admin_dir's startup directory.

Logging in is now automatically handled, and when there is a server cutover, the overseer_agent will provide the new SP
endpoint information for the active server and the FLAdminAPI will reauthenticate so the commands will be sent to the
new active server.

``logout()`` function can be called to log out. Both ``login()`` and ``logout()`` are
inherited from AdminAPI.

.. code:: python

    api.logout()

After using FLAdminAPI, the overseer_agent must be cleaned up with a call to:

.. code:: python

    api.overseer_agent.end()

Usage
-----
See the example scripts ``run_fl.py`` in `CIFAR-10 <https://github.com/NVIDIA/NVFlare/tree/main/examples/cifar10>`_ for
an example of how to use the FLAdminAPI with FLAdminAPIRunner.

You can use the example as inspiration to write your own code using the FLAdminAPI to operate your FL system.

Arguments and targets
---------------------
The arguments required for each FLAdminAPI call are specified in :class:`FLAdminAPI<nvflare.fuel.hci.client.fl_admin_api_spec.FLAdminAPISpec>`.

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
debugging).

Implementation Notes
--------------------
FLAdminAPI uses the underlying AdminAPI's ``do_command()`` function to submit commands to the server, and you
can also use this function directly for functions that are not wrapped in an FLAdminAPI function. Returns from AdminAPI
are included in the FLAdminAPI reply under the "raw" key for some calls and error conditions.

Additional and Complex Commands
-------------------------------
The functions ``wait_until_server_status()``, ``wait_until_client_status()``, and ``wait_until_server_stats()`` are
included with the FLAdminAPI in NVIDIA FLARE as examples of useful functions that can be built with other calls in a
loop with logic. These examples wait until the provided callback returns True, with the option to specify a timeout and
interval to check the status or stats. There is a default callback to evaluate the reply in the included functions, and
additional kwargs passed in will be available to the callback. Custom callbacks can be provided to add logic to handle
checking for other conditions. For these example functions, a timeout should be set in case there are any error
conditions that result in the system being stuck in a state where the callback never returns True.

You can use the source code of these function as inspiration to create your own functions or logic that makes use of
other FLAdminAPI calls.

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
