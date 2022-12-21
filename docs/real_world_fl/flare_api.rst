.. _flare_api:

FLARE API
=========

:mod:`FLARE API<nvflare.fuel.flare_api.flare_api>` is the FLAdminAPI redesigned for a better user experience in version 2.3. It is a
wrapper for admin commands that can be issued to the FL server like the FLAdminAPI, and you can use a provisioned admin
client's certs and keys to initialize a :class:`Session<nvflare.fuel.flare_api.flare_api.Session>` to use the commands of the API.

Initialization and Usage
------------------------
Initialize the FLARE API with :func:`new_secure_session<nvflare.fuel.flare_api.flare_api.new_secure_session>` by providing
the username and the path to the startup kit folder of the provisioned user containing the startup folder with the admin client's
certs and keys::

.. code:: python

    from nvflare.fuel.flare_api.flare_api import new_secure_session

    sess = new_secure_session(
        "super@nvidia.com",
        "/workspace/example_project/prod_00/super@nvidia.com"
    )

Logging in is automatically handled, and commands can be executed with the session object returned (``sess`` in the preceding code block).

Using the FLARE API should be similar to the previous FLAdminAPI but simpler. The return structure is no longer an object with a status and a
dictionary of details, but can be a string or even nothing depending on the command. Instead of having a status with an error, FLARE API now
will raise an exception, and the handling of these exceptions will be the responsibility of the code using the FLARE API.

There should now be no need to have wrappers around the commands since what is being returned is much simpler, for example, ``submit_job``
now returns the job id as a string if the job is accepted by the system. See the details of what each command returns in the docstrings at:
:mod:`FLARE API<nvflare.fuel.flare_api.flare_api>`.

Implementation Notes
--------------------
Like with FLAdminAPI previously, :class:`AdminAPI<nvflare.fuel.hci.client.api.AdminAPI>` is used to connect and submit commands to the server.

There is no more ``logout()``, instead, use ``close()`` end the session. One common pattern of usage may be to have the code using the session
to execute commands inside a try block and then close
the session in a finally clause::

.. code:: python

    try:
        print(sess.get_system_info())
        job_id = sess.submit_job("/workspace/locataion_of_jobs/job1")
        print(job_id + " was submitted")
        # monitor_job() waits until the job is done, see the section about it below for details
        sess.monitor_job(job_id)
        print("job done!")
    finally:
        sess.close()


.. note::

    The session monitor may take a bit of time to close when ``close()`` is called.

Additional and Complex Commands
-------------------------------
With a ``job_id`` for example after submit_job in the code block above, here are some examples of other commands that
can be run::

.. code:: python

    # get job meta dictionary with job info
    job_meta_dict = sess.get_job_meta(job_id)

    # submit a copy of an existing job with clone_job
    new_job_id = sess.clone_job(job_id)
    print(new_job_id + " was submitted as a clone of " + job_id)

Monitor Job
^^^^^^^^^^^
By default, ``monitor_job()`` waits until the job specified as the first argument is finished, but it can be used in
more custom ways by providing additional args including your own callback with custom code to be called after each
status poll.
