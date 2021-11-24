#######################################################################################
How to create the self-signed SSL Certificate Authority and server-clients certificates
#######################################################################################

If you are using the automatically generated startup kits created by the provisioning tool as described in the
:ref:`user_guide/provisioning_tool:Provisioning in NVIDIA FLARE`, the following would have already been taken care of at the
creation of the startup kits. The code that generates the startup kits automatically creates certificates for the server
and clients and puts them all in their default expected locations.

.. highlight:: none

If you are not using the startup kits and want to create certificates yourself manually, the following is an example of
how that could be done::

    ## 1. Server root key and certificate

    ## 1.1 Server creates the root private key
    `openssl genrsa -out rootCA.key 2048`   Or
    `openssl genrsa -des3 -out rootCA.key 2048` (with password)

    ## 1.2 Server creates the self-signed root certificate
    `openssl req -x509 -new -nodes -key rootCA.key -sha256 -days 1024 -out rootCA.pem`

    ## 2. Server private key and CSR

    ## 2.1 Server creates private key
    `openssl genrsa -out server.key 2048`
    ## 2.2 Server creates certificate signing request (CSR)
    `openssl req -new -key server.key -out server.csr`
    ## 2.3 Server signs the CSR using the root certificate rootCA.pem
    `openssl x509 -req -in server.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out server.crt -days 500 -sha256`

    ## 3. Client certificate

    ## Important: client must input a common name on command below
    ## 3.1 Client creates private key
    `openssl genrsa -out client3.key 2048`
    ## 3.2 Client creates CSR
    `openssl req -new -key client3.key -out client3.csr`

    ## 4. Sign the CSR using the root certificate and place signed certificate in client's config path

    ## 4.1 Server runs this after getting client3.csr
    `openssl x509 -req -in client3.csr -CA rootCA.pem -CAkey rootCA.key -CAcreateserial -out client3.crt -days 500 -sha256`

    ## 4.2 Server gives client client3.crt to place in the client's config path
