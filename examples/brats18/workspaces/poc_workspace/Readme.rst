###############################
How to use nvFlare POC commands
###############################

nvFlare POC commands enables to set up and run the FL training without the provision steps. The FL server communicates with the clients and admin without HTTPS encryption. There's no
SSL certificates required on server, or the client and admin. You can run the FL server, client and admin system components on
the same machine, or from different machines.


#. To start FL server

    $ cd poc/server

    $ startup/start.sh

    or

    $ startup/start.sh server_host(name or IP)

#. To start FL client (for each client, make a copy of the poc/client folder)

    $ cd poc/site-1

    $ startup/start.sh

    or 
    
    $ startup/start.sh server_host(name or IP)

    or
    
    $ startup/start.sh server_host(name or IP) client_name

#. To start Admin (use the default user "admin/admin")

    $ cd poc/admin

    $ startup/fl_admin.sh

    or

    $ startup/fl_admin.sh server_host(name or IP)

