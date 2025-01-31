.. _reliable_xgboost_timeout:

############################################
Reliable Federated XGBoost Timeout Mechanism
############################################

NVFlare introduces a tightly-coupled integration between XGBoost and NVFlare.
NVFlare implements the :class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>`
mechanism to make XGBoost's server/client interactions more robust over
unstable internet connections.

Unstable internet connection is the situation where the connections between
the communication endpoints have random disconnects/reconnects and unstable speed.
It is not meant to be an extended internet outage.

ReliableMessage does not mean guaranteed delivery.
It only means that it will try its best to deliver the message to the peer.
If one attempt fails, it will keep trying until either the message is
successfully delivered or a specified "transaction timeout" is reached.

*****************
Timeout Mechanism
*****************

In runtime, the FLARE System is configured with a few important timeout parameters.

ReliableMessage Timeout
=======================

There are two timeout values to control the behavior of ReliableMessage (RM).

Per-Message Timeout
-------------------

Essentially RM tries to resend the message until delivered successfully.
Each resend of the message requires a timeout value.
This value should be defined based on the message size, overall network speed,
and the amount of time needed to process the message in a normal situation.
For example, if an XGBoost message takes no more than 5 seconds to be
sent, processed, and replied.
The per-message timeout should be set to 5 seconds.

.. note::

    Note that the initial XGBoost message may take more than 100 seconds,
    depending on the dataset size.

Transaction Timeout
-------------------

This value defines how long you want RM to keep retrying until done, in case
of unstable connection.
This value should be defined based on the overall stability of the connection,
nature of the connection, and how quickly the connection is restored.
For occasional connection glitches, this value shouldn't have to be too big
(e.g. 20 seconds).
However if the outage is long (say 60 seconds or longer), then this value
should be big enough.

.. note::

    Note that even if you think the connection is restored (e.g. replugged
    the internet cable or reactivated WIFI), the underlying connection
    layer may take much longer to actually restore connections (e.g. up to
    a few minutes)!

.. note::

    Note: if the transaction timeout is <= per-message timeout, then the
    message will be sent through simple messaging - no retry will be done
    in case of failure.

XGBoost Client Operation Timeout
================================

To prevent a XGBoost client from running forever, the XGBoost/FLARE
integration lets you define a parameter (max_client_op_interval) on the
server side to control the max amount of time permitted for a client to be
silent (i.e. no messages sent to the server).
The default value of this parameter is 900 seconds, meaning that if no XGB
message is received from the client for over 900 seconds, then that client
is considered dead, and the whole job is aborted.

***************************
Configure Timeouts Properly
***************************

These timeout values are related. For example, if the transaction timeout
is greater than the server timeout, then it won't be that effective since
the server will treat the client to be dead once the server timeout is reached
anyway. Similarly, it does not make sense to have transaction timeout > XGBoost
client op timeout.

In general, follow this rule:

Per-Message Timeout < Transaction Timeout < XGBoost Client Operation Timeout
