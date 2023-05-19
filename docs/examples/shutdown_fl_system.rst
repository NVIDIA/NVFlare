Shutdown FL system
^^^^^^^^^^^^^^^^^^

Once the FL run is complete and the server has successfully aggregated the client's results after all the rounds, and
cross site model evaluation is finished, run the following commands in the fl_admin to shutdown the system (while
inputting ``admin`` when prompted with password):

.. code-block:: shell

    > shutdown client
    > shutdown server
    > bye