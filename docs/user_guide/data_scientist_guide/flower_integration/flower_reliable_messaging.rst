******************
Reliable Messaging
******************

The interaction between the FLARE Clients and Server is through reliable messaging. 
First, the requester tries to send the request to the peer. If it fails to send it, it will retry a moment later.
This process keeps repeating until the request is sent successfully or the amount of time has passed (which will
cause the job to abort).

Secondly, once the request is sent, the requester waits for the response. Once the peer finishes processing, it
sends the result to the requester immediately (which could be successful or unsuccessful). At the same time, the
requester repeatedly sends queries to get the result from the peer, until the result is received or the max amount
of time has passed (which will cause the job to abort). The result could be received in one of the following ways:

    - The result is received from the response message sent by the peer when it finishes the processing
    - The result is received from the response to the query message of the requester

For details of :class:`ReliableMessage<nvflare.apis.utils.reliable_message.ReliableMessage>`,
see :ref:`ReliableMessage Timeout <reliable_xgboost_timeout>`.
