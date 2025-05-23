.. _migrating_to_flare_api:

Migrating to FLARE API from FLAdminAPI
======================================

:mod:`FLARE API<nvflare.fuel.flare_api.flare_api>` is the :ref:`fladmin_api` redesigned for a better user experience in version 2.3.
Like the FLAdminAPI, the FLARE API is a wrapper for admin commands that can be issued to the FL server, and you can use a provisioned admin
client's certs and keys to initialize a :class:`Session<nvflare.fuel.flare_api.flare_api.Session>` to use the commands of the API.

This page goes through all the differences to help you migrate from using the FLAdminAPI to the new FLARE API. Note that only a subset of the
frequently used commands have been implemented in the FLARE API so far, but you can still execute any legacy command if you want.

.. _migrating_to_flare_api_initialization:

Migrating API Initialization
----------------------------
Initialization of the FLAdminAPI was cumbersome due to all the necessary arguments including paths to certs and an overseer_agent, so an
:class:`FLAdminAPIRunner<nvflare.fuel.hci.client.fl_admin_api_runner.FLAdminAPIRunner>` was used for initializing the FLAdminAPI
with the username of the admin user and the path to the admin startup kit directory.

Initializing the FLAdminAPI:

.. code-block:: python

    api_instance = FLAdminAPI(
        ca_cert="/workspace/example_project/prod_00/super@nvidia.com/startup/rootCA.pem",
        client_cert="/workspace/example_project/prod_00/super@nvidia.com/startup/client.crt",
        client_key="/workspace/example_project/prod_00/super@nvidia.com/startup/client.key",
        upload_dir="/workspace/example_project/prod_00/super@nvidia.com/transfer",
        download_dir="/workspace/example_project/prod_00/super@nvidia.com/transfer",
        overseer_agent=overseer_agent,
        user_name="super@nvidia.com"
    )

Initializing the FLAdminAPIRunner, which initializes FLAdminAPI with the values in fed_admin.json of the startup kit in the provided admin_dir:

.. code-block:: python

    runner = FLAdminAPIRunner(  
        username="super@nvidia.com",
        admin_dir="/workspace/example_project/prod_00/super@nvidia.com"
    )

:ref:`flare_api_initialization` is similar to :class:`FLAdminAPIRunner<nvflare.fuel.hci.client.fl_admin_api_runner.FLAdminAPIRunner>`
with :func:`new_secure_session<nvflare.fuel.flare_api.flare_api.new_secure_session>` taking two required arguments of
the username and the path to the root admin directory containing the startup folder with the admin client's
certs and keys:

.. code-block:: python

    from nvflare.fuel.flare_api.flare_api import new_secure_session

    sess = new_secure_session(
        username="super@nvidia.com",
        startup_kit_location="/workspace/example_project/prod_00/super@nvidia.com"
    )


Logging in is automatically handled, and commands can be executed with the session object returned (``sess`` in the preceding code block).
This is in contrast to :ref:`fladmin_api` where the command was issued through the API object itself, or in the case of :class:`FLAdminAPIRunner<nvflare.fuel.hci.client.fl_admin_api_runner.FLAdminAPIRunner>`,
``self.api`` (in the code blocks below, ``runner.api`` is used for the FLAdminAPI).


General Notes on Migrating to FLARE API
---------------------------------------

Return Structure
^^^^^^^^^^^^^^^^
The return structure for FLAdminAPI commands was an ``FLAdminAPIResponse`` object that contained the status, details, and raw response from the server.
This required parsing the response to get the status or other information to then use or output. The FLARE API no longer returns an object with a
status and a dictionary of details, but the response depends on the command and is greatly simplified. See the details of what each command returns below
or in the docstrings at: :mod:`FLARE API<nvflare.fuel.flare_api.flare_api>`.

FLARE API Now Raises Exceptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instead of having a status with an error that needs to be parsed in FLAdminAPI, FLARE API will now raise an exception if there is an error or
something unexpected happens, and the handling of these exceptions will be the responsibility of the code using the FLARE API. This means that in general,
there is no more need for something like ``api_command_wrapper()`` that parsed the responses from FLAdminAPI.

Closing the Session
^^^^^^^^^^^^^^^^^^^
For FLARE API, use ``close()`` to end the session. It is ideal to execute commands with a session inside a try block with ``close()`` in a ``finally`` block.
For details, see :ref:`flare_api_implementation_notes`.


.. _migrating_fladminapi_commands_to_flare_api:

Migrating FLAdminAPI Commands to FLARE API
------------------------------------------
This section has a summary of the commands then goes through each command and shows examples of the usage and output from before with FLAdminAPI
and the new way with FLARE API.

.. csv-table::
    :header: FLAdminAPI,FLARE API,Version Added,Notes
    :widths: 15, 15, 30, 30

    check_status,get_system_info,2.3.0,Simplified and reformatted output (see below for details)
    submit_job,submit_job,2.3.0,Simplified output (see below for details)
    list_job,list_job,2.3.0,Simplified output (see below for details)
    wait_until_server_status,monitor_job,2.3.0,Changed the arg names and function (see below for details)
    download_job,download_job_result,2.3.0,Simplified output (see below for details)
    clone_job,clone_job,2.3.0,Simplified output (see below for details)
    abort_job,abort_job,2.3.0,Simplified output (see below for details)
    delete_job,delete_job,2.3.0,Simplified output (see below for details)
    check_status,get_client_job_status,2.4.0,only for client
    restart,restart,2.4.0,
    shutdown,shutdown,2.4.0,
    set_timeout,set_timeout,2.4.0,changed to session-based
    list_sp,list_sp,2.4.0,
    get_active_sp,get_active_sp,2.4.0,
    promote_sp,promote_sp,2.4.0,
    get_available_apps_to_upload,get_available_apps_to_upload,2.4.0,
    shutdown_system,shutdown_system,2.4.0,
    ls_target,ls_target,2.4.0,
    cat_target,cat_target,2.4.0,
    ,tail_target,2.4.0,added for consistency
    tail_target_log,tail_target_log,2.4.0,
    ,head_target,2.4.0,new
    ,head_target_log,2.4.0,new
    grep_target,grep_target,2.4.0,
    get_working_directory,get_working_directory,2.4.0,
    show_stats,show_stats,2.4.0,return structure changed
    show_errors,show_errors,2.4.0,return structure changed
    reset_errors,reset_errors,2.4.0,
    get_connected_client_list,get_connected_client_list,2.4.0,
    abort,,2.4.0,obsolete
    remove_client,,2.4.0,not exposed

Get System Info from Check Status
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting the system information before with the FLAdminAPI was primarily done through the ``check_status()`` command:

.. code-block:: python

    from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType

    api_command_wrapper(runner.api.check_status(TargetType.SERVER))

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {<FLDetailKey.SERVER_ENGINE_STATUS: 'server_engine_status'>: 'stopped',
    <FLDetailKey.STATUS_TABLE: 'status_table'>: [['CLIENT',
        'TOKEN',
        'LAST CONNECT TIME'],
    ['site_a',
        '32ebdf1c-b51b-4eb3-ae49-4ac488a2aaa1',
        'Thu Jan 26 15:13:12 2023'],
    ['site_b',
        '4bbfb243-9ae1-4339-9e6d-750092ebc240',
        'Thu Jan 26 15:13:12 2023']],
    <FLDetailKey.REGISTERED_CLIENTS: 'registered_clients'>: 2},
    'raw': {'time': '2023-01-26 15:13:25.652993',
    'data': [{'type': 'string', 'data': 'Engine status: stopped'},
    {'type': 'table', 'rows': [['JOB_ID', 'APP NAME']]},
    {'type': 'string', 'data': 'Registered clients: 2 '},
    {'type': 'table',
        'rows': [['CLIENT', 'TOKEN', 'LAST CONNECT TIME'],
        ['site_a',
        '32ebdf1c-b51b-4eb3-ae49-4ac488a2aaa1',
        'Thu Jan 26 15:13:12 2023'],
        ['site_b',
        '4bbfb243-9ae1-4339-9e6d-750092ebc240',
        'Thu Jan 26 15:13:12 2023']]}],
    'meta': {'status': 'ok',
    'info': '',
    'server_status': 'stopped',
    'server_start_time': 1674763921.3592467,
    'jobs': [],
    'clients': [{'client_name': 'site_a',
        'client_last_conn_time': 1674763992.4529057},
        {'client_name': 'site_b', 'client_last_conn_time': 1674763992.4763987}]},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}


With the FLARE API, the new command ``get_system_info()`` returns a SystemInfo object consisting of server_info
(server status and start time), client_info (each connected client and the last connect time for that client), and job_info
(the list of current jobs with the job_id and app_name).

.. code-block:: python

    sess.get_system_info()

Calling print on the :class:`SystemInfo<nvflare.fuel.flare_api.api_spec.SystemInfo>` object will give a result like the following,
or you can access the server_info, client_info, and job_info variables to access the data within.

.. code-block:: bash

    SystemInfo
    server_info: status: stopped, start_time: Thu Jan 26 15:12:01 2023
    client_info: 
    site_a(last_connect_time: Thu Jan 26 15:12:42 2023)
    site_b(last_connect_time: Thu Jan 26 15:12:42 2023)
    job_info:
    job_id: 44d32a5f-9766-44b6-aef5-7ed9fd168335
    app_name: hello-numpy-sag


Submit Job
^^^^^^^^^^
The ``submit_job()`` command for the FLAdminAPI and FLARE API are very similar. The necessary argument is the same for both, the
path to the job to submit as a string. For ``submit_job()`` with FLAdminAPI:

.. code-block:: python

    path_to_example_job = "/workspace/NVFlare/examples/hello-world/hello-numpy-sag"
    runner.api.submit_job(path_to_example_job)

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {'message': 'Submitted job: 5d0eaa30-6936-4044-918e-cd9c3f5edf9b',
    'job_id': '5d0eaa30-6936-4044-918e-cd9c3f5edf9b'},
    'raw': {'time': '2023-01-26 15:30:35.260527',
    'data': [{'type': 'string',
        'data': 'Submitted job: 5d0eaa30-6936-4044-918e-cd9c3f5edf9b'},
    {'type': 'success', 'data': ''}],
    'meta': {'status': 'ok',
    'info': '',
    'job_id': '5d0eaa30-6936-4044-918e-cd9c3f5edf9b'},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}


With the FLARE API, ``submit_job()`` returns the job_id of the job if it is successfully submitted so you can save that
value to use later.

.. code-block:: python

    path_to_example_job = "/workspace/NVFlare/examples/hello-world/hello-numpy-sag"
    job_id = sess.submit_job(path_to_example_job)
    print(job_id + " was submitted")

.. code-block:: bash

    5d0eaa30-6936-4044-918e-cd9c3f5edf9b was submitted


List Jobs
^^^^^^^^^^
The ``list_jobs()`` command for FLAdminAPI took an optional argument of a string for the options, and with the FLARE API, the options are
set as boolean values. For ``list_jobs()`` with FLAdminAPI:

.. code-block:: python

    runner.api.list_jobs()
    # runner.api.list_jobs("-a -d")

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': [['JOB ID', 'NAME', 'STATUS', 'SUBMIT TIME', 'RUN DURATION'],
    ['5d0eaa30-6936-4044-918e-cd9c3f5edf9b',
    'hello-numpy-sag',
    'FINISHED:COMPLETED',
    '2023-01-26T15:30:35.262048-05:00',
    '0:00:48.170128']],
    'raw': {'time': '2023-01-26 15:47:23.091621',
    'data': [{'type': 'table',
        'rows': [['JOB ID', 'NAME', 'STATUS', 'SUBMIT TIME', 'RUN DURATION'],
        ['5d0eaa30-6936-4044-918e-cd9c3f5edf9b',
        'hello-numpy-sag',
        'FINISHED:COMPLETED',
        '2023-01-26T15:30:35.262048-05:00',
        '0:00:48.170128']]},
    {'type': 'success', 'data': ''}],
    'meta': {'jobs': [{'job_id': '5d0eaa30-6936-4044-918e-cd9c3f5edf9b',
        'job_name': 'hello-numpy-sag',
        'status': 'FINISHED:COMPLETED',
        'submit_time': '2023-01-26T15:30:35.262048-05:00',
        'duration': '0:00:48.170128'}],
    'status': 'ok',
    'info': ''},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}


With the FLARE API, ``list_job()``:

.. code-block:: python

    list_jobs_output = sess.list_jobs()
    print(list_jobs_output)
    # list_jobs_output_detailed_all = sess.list_jobs(detailed=True, all=True)
    # print(list_jobs_output_detailed_all)

.. code-block:: bash

    [{'job_id': '9382ff9e-eb7e-4e0d-9a8e-78c82747b5ac', 'job_name': 'hello-numpy-sag', 'status': 'RUNNING', 'submit_time': '2023-01-26T15:56:30.188836-05:00', 'duration': '0:00:32.686275'}]


Monitor Job from Wait Until
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the FLAdminAPI, there were ``wait_until_server_status()`` and ``wait_until_client_status()`` that you could use to
monitor the status of the training:

.. code-block:: python

    runner.api.wait_until_server_status()

By default, the ``wait_until`` functions for FLAdminAPI waited until the server engine status was stopped or the clients no longer had
any active jobs before returning a status of "SUCCESS".

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>}

With the FLARE API, ``monitor_job()`` provides a similar function but takes a required argument of a job_id to continuously retrieve the
job meta information for the job status until that job is done.

.. code-block:: python

    sess.monitor_job(job_id)

.. code-block:: bash

    <MonitorReturnCode.JOB_FINISHED: 0>

The additional optional arguments have been slightly modified with ``interval`` becoming ``poll_interval`` and type float instead of int,
``timeout`` remaining the same name but type float instead of int, and ``callback`` to ``cb``.

The ``monitor_job()`` command of the FLARE API is intended to be customizable with callbacks, see :ref:`flare_api_monitor_job` for more details.


Download Job Result from Download Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``download_job()`` command for FLAdminAPI has been renamed to ``download_job_result()``. It took a required argument of job_id as a string,
and this remains the same for the FLARE API. The behavior of the command remains the same, with the output being simplified just to the path
to the downloaded job. With FLAdminAPI:

.. code-block:: python

    runner.api.download_job(job_id)

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {'message': 'Download to dir /workspace/workspace/hello-example/prod_00/admin@nvidia.com/transfer'},
    'raw': {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': 'Download to dir /workspace/workspace/hello-example/prod_00/admin@nvidia.com/transfer',
    'meta': {'status': 'ok',
    'info': '',
    'job_id': '5d0eaa30-6936-4044-918e-cd9c3f5edf9b'}}}

With the FLARE API, ``download_job_result()``:

.. code-block:: python

    sess.download_job_result(job_id)

.. code-block:: bash

    '/workspace/workspace/hello-example/prod_00/admin@nvidia.com/transfer/5d0eaa30-6936-4044-918e-cd9c3f5edf9b'


Clone Job
^^^^^^^^^
The usage for the ``clone_job()`` command is the same for FLAdminAPI and the FLARE API with just the job_id as a string as the required argument.
The behavior of the command remains the same, with the output being simplified just to the job_id of the newly cloned job. With FLAdminAPI:

.. code-block:: python

    runner.api.clone_job(job_id)

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {'message': 'Cloned job 5d0eaa30-6936-4044-918e-cd9c3f5edf9b as: 4a2cf195-314d-4476-9ea5-c69bed397e3a',
    'job_id': '4a2cf195-314d-4476-9ea5-c69bed397e3a'},
    'raw': {'time': '2023-01-25 15:08:40.235304',
    'data': [{'type': 'string',
        'data': 'Cloned job 5d0eaa30-6936-4044-918e-cd9c3f5edf9b as: 4a2cf195-314d-4476-9ea5-c69bed397e3a'},
    {'type': 'success', 'data': ''}],
    'meta': {'status': 'ok',
    'info': '',
    'job_id': '4a2cf195-314d-4476-9ea5-c69bed397e3a'},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}

With the FLARE API, ``clone_job()``:

.. code-block:: python

    sess.clone_job(job_id)

.. code-block:: bash

    '4a2cf195-314d-4476-9ea5-c69bed397e3a'


Abort Job
^^^^^^^^^
The ``abort_job()`` command is the same for FLAdminAPI and the FLARE API with just the job_id as a string as the required argument.
The behavior of the command remains the same, with the output being simplified to None. With FLAdminAPI:

.. code-block:: python

    runner.api.abort_job(job_id)

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {'message': 'Abort signal has been sent to the server app.'},
    'raw': {'time': '2023-01-26 16:59:32.980711',
    'data': [{'type': 'string',
        'data': 'Abort signal has been sent to the server app.'},
    {'type': 'success', 'data': ''}],
    'meta': {'status': 'ok', 'info': ''},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}

With the FLARE API, ``abort_job()``:

.. code-block:: python

    sess.abort_job(job_id)

.. code-block:: bash

    None


Delete Job
^^^^^^^^^^
The ``delete_job()`` command is the same for FLAdminAPI and the FLARE API with just the job_id as a string as the required argument.
The behavior of the command remains the same, with the output being simplified to nothing. With FLAdminAPI:

.. code-block:: python

    runner.api.delete_job(job_id)

.. code-block:: bash

    {'status': <APIStatus.SUCCESS: 'SUCCESS'>,
    'details': {'message': 'Job 4a2cf195-314d-4476-9ea5-c69bed397e3a deleted.'},
    'raw': {'time': '2023-01-26 17:02:12.812807',
    'data': [{'type': 'string',
        'data': 'Job 4a2cf195-314d-4476-9ea5-c69bed397e3a deleted.'},
    {'type': 'success', 'data': ''}],
    'meta': {'status': 'ok', 'info': ''},
    'status': <APIStatus.SUCCESS: 'SUCCESS'>}}

With the FLARE API, ``delete_job()``:

.. code-block:: python

    sess.delete_job(job_id)

Migrating All Other FLAdminAPI Commands to FLARE API
----------------------------------------------------
The remaining FLAdminAPI commands have been added to the FLARE API in 2.4.0.
For more details, see the notes in the table above, and the :mod:`FLARE API<nvflare.fuel.flare_api.flare_api>` definitions.
