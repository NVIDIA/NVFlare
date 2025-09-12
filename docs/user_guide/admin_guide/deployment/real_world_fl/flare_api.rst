.. _flare_api:

FLARE API
=========

:mod:`FLARE API<nvflare.fuel.flare_api.flare_api>` is the FLAdminAPI redesigned for a better user experience in version 2.3. It is a
wrapper for admin commands that can be issued to the FL server like the FLAdminAPI, and you can use a provisioned admin
client's certs and keys to initialize a :class:`Session<nvflare.fuel.flare_api.flare_api.Session>` to use the commands of the API.

.. _flare_api_initialization:

Initialization and Usage
------------------------
Initialize the FLARE API with :func:`new_secure_session<nvflare.fuel.flare_api.flare_api.new_secure_session>` by providing
the username and the path to the startup kit folder of the provisioned user containing the startup folder with the admin client's
certs and keys:

.. code-block:: python

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

.. _flare_api_implementation_notes:

Implementation Notes
--------------------
Like with FLAdminAPI previously, :class:`AdminAPI<nvflare.fuel.hci.client.api.AdminAPI>` is used to connect and submit commands to the server.

There is no more ``logout()``, instead, use ``close()`` to end the session. One common pattern of usage may be to have the code using the session
to execute commands inside a try block and then close
the session in a finally clause:

.. code-block:: python

    try:
        print(sess.get_system_info())
        job_id = sess.submit_job("/workspace/location_of_jobs/job1")
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
can be run:

.. code-block:: python

    # get job meta dictionary with job info
    job_meta_dict = sess.get_job_meta(job_id)

    # submit a copy of an existing job with clone_job
    new_job_id = sess.clone_job(job_id)
    print(new_job_id + " was submitted as a clone of " + job_id)

.. _flare_api_monitor_job:

Monitor Job
^^^^^^^^^^^
By default, like in the most basic usage above in :ref:`flare_api_implementation_notes`, ``monitor_job()`` waits until
the job specified as the first argument is finished, but it can be used in more custom ways by providing additional args
including your own callback with custom code to be called after each status poll. The following is the API spec for
monitor_job:

.. code-block:: python

    def monitor_job(
        self, job_id: str, timeout: int = 0, poll_interval: float = 2.0, cb=None, *cb_args, **cb_kwargs
    ) -> MonitorReturnCode:
        """Monitor the job progress until one of the conditions occurs:
         - job is done
         - timeout
         - the status_cb returns False

        Args:
            job_id: the job to be monitored
            timeout: how long to monitor. If 0, never time out.
            poll_interval: how often to poll job status
            cb: if provided, callback to be called after each poll

        Returns: a MonitorReturnCode

        Every time the cb is called, it must return a bool indicating whether the monitor
        should continue. If False, this method ends.

        """

Only the first argument is required, but with additional args, you can customize ``monitor_job()`` to do almost
anything you want to do. The following is an example where you can see the usage of a sample_cb and cb_kwargs.
This callback always returns True, keeping the default behavior of ``monitor_job()`` of waiting until the job specified
as the first argument is finished, but you can customize this to behave as you want.

.. code-block:: python

    def sample_cb(
        session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs
    ) -> bool:
        if job_meta["status"] == "RUNNING":
            if cb_kwargs["cb_run_counter"]["count"] < 3:
                print(job_meta)
                print(cb_kwargs["cb_run_counter"])
            else:
                print(".", end="")
        else:
            print("\n" + str(job_meta))
        
        cb_kwargs["cb_run_counter"]["count"] += 1
        return True

    # Calling monitor_job with the sample_cb above and a cb_kwarg
    sess.monitor_job(job_id, cb=sample_cb, cb_run_counter={"count":0})
