openssl req -nodes -x509 -newkey rsa:2048 -keyout ca.key -out ca.crt -subj "/C=US/ST=California/L=Santa Clara/O=NVIDIA/OU=NVFlare/CN=NVFlare/emailAddress=zhihongz@nvidia.com"


# Generate server cert to be signed
openssl req -nodes -newkey rsa:2048 -keyout server.key -out server.csr -subj "/C=US/ST=California/L=Santa Clara/O=NVIDIA/OU=NVFlare/CN=localhost/emailAddress=zhihongz@nvidia.com"

# Sign the server cert
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt

# Generate client cert to be signed
openssl req -nodes -newkey rsa:2048 -keyout client.key -out client.csr -subj "/C=US/ST=California/L=Santa Clara/O=NVIDIA/OU=NVFlare/CN=nvflare-client/emailAddress=zhihongz@nvidia.com"

# Sign the client cert
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAserial ca.srl -out client.crt

