.. _server_port_consolidation:

FL Server Port Consolidation
============================

Historically, Flare’s FL Server requires two communication port numbers to be open to the public. One port is used for FL Client/Server communication, another is for Admin Client/Server communication. For customers that port numbers are strictly managed, getting an extra port number could be challenging.

Flare 2.7 consolidates port number requirement to one: the same port number can be used for both types of communication!

For some customers, it may still be desirable to use different port numbers because they can be managed under different network security policies. To accommodate such customers, the system can still be provisioned to use two different port numbers for admin/server and client/server communications.

**Connection Example Illustration**

The following diagrams illustrate the two different connection and authentication mechanisms
enabled by the single port, TLS, bring your own connection features.

.. image:: resources/flare_byocc.png
    :height: 300px


Detailed Changes
----------------

In previous versions, the Admin Client communicates with the server via a TCP connection. This is handled separately from the Cellnet technology that is used for FL Client/Server communication.

Flare 2.7 modified the Admin Client to also use the Cellnet technology. This made it possible for the admin client to talk to the server using the same port number that is used for FL client/server communications.

All the changes should be transparent to the end user - the user experience for the admin client remains the same.

Port Number Provision
---------------------

The FL Port number is specified with the “fed_learn_port” property in the project’s provision config file (e.g. project.yml). See example below.

.. code-block:: yaml

   participants:
    # change example.com to the FQDN of the server
    - name: server
      type: server
      org: nvidia
      use_aio: false
      fed_learn_port: 8002
      host_names: [localhost, 127.0.0.1]
      default_host: localhost

If the property is not explicitly specified, it will be defaulted to 8002. By default, the fed_learn_port is also used as the admin_port. But if you want, you can also specify a different port number using the “admin_port” property.

.. code-block:: yaml

   participants:
    # change example.com to the FQDN of the server
    - name: server
      type: server
      org: nvidia
      use_aio: false
      fed_learn_port: 8002
      admin_port: 8003
      host_names: [localhost, 127.0.0.1]
      default_host: localhost

Admin Client Configuration
--------------------------

Once provisioned, an admin user will receive a startup kit, which is to be used for the user to connect to the Flare Server using the admin client (or Flare API).

The “startup” folder in the kit contains essential configuration information that must not be modified by the user. If the file is modified, the admin client will detect it and won’t connect to the server.

In the “local” folder in the kit there is the “resources.json.default” file that contains configuration parameters that the user can change.

.. code-block:: json

   {
    "format_version": 1,
    "admin": {
      "idle_timeout": 900.0,
      "login_timeout": 10.0,
      "with_debug": false,
      "authenticate_msg_timeout": 2.0,
      "prompt": "> "
    }
   }

The user can edit this file and set the parameters to fit to his/her local environment more properly.

Idle Timeout
------------

For security, the admin client automatically shuts down when being idle too long. The idle_timeout parameter specifies how long the client is allowed to be idle before automatic shutdown.

The default value is 900 seconds.

Login Timeout
-------------

When the admin client is started, it will try to log in. However, the FL Server may or may not be available at the time login. The admin client will keep trying until a preset timeout is reached.

The login_timeout parameter specifies how long you want to try to log in before quitting. The default value is 10 seconds.

Authentication Message Timeout
------------------------------

One of the steps for login is authentication. Multiple messages are used between the admin client and the FL server to authenticate them to each other.

The authenticate_msg_timeout parameter specifies the timeout value for these messages. The default value is 2 seconds.

You should consider increasing the value only if your local network is slow.

Enable Debug
------------

Normally the admin client runs without printing debugging information. In case you run into errors, you may enable debugging to have detailed technical information printed.

To enable debugging, set the with_debug parameter to true.

Command Prompt
--------------

When the admin client is started, it displays a prompt character for you to enter commands. This character is specified with the prompt parameter. You can change the prompt character to whatever you like.

Command Timeout
---------------

Commands are sent to the FL Server for execution through messages. The default timeout for each message is 5 seconds. In case your network is slow, you may want to increase it to a bigger value.

You can change command timeout:

- If you are running the admin client, issue the “timeout <value>” command;
- Call `sess.set_timeout(value)` method when using Flare API.





