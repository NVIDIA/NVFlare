.. _kubernetes_ingress_tls:

########################################
Kubernetes Ingress and TLS
########################################

You can expose a FLARE server running in Kubernetes with TLS in either of these ways:

- **TLS at ingress** — The ingress controller terminates TLS (for example on port 443). The server runs with ``connection_security: clear`` behind the ingress. Clients and admin use the ingress hostname on port 443.
- **TCP passthrough (server TLS/mTLS)** — The ingress forwards TCP unchanged. The FLARE server terminates TLS or mTLS on a single port (for example 8002) for both FL and admin. Clients connect to that hostname and port; encryption is end-to-end to the server.

For connection-security modes (clear, TLS, mTLS), see :ref:`communication_security`. For common failures and workarounds, see :ref:`k8s_ingress_tls_troubleshooting`.

.. note::
   **One server endpoint per Ingress (TLS at ingress).** If you run NVFlare in Kubernetes and use **TLS at ingress**, design for **one** FLARE server Service behind that hostname (one public entry point for that federation). Do not route multiple independent FL server deployments through the same Ingress host/path unless you deliberately split traffic (for example separate hostnames). If you need a **dedicated open TCP port** and server-terminated TLS or mTLS—or a topology that does not fit L7 Ingress—use **TCP passthrough** (below) instead of HTTP(S) Ingress.

***************************
TLS at ingress
***************************

#. **Configure the project** — Set the server to ``connection_security: clear``. Set ``host_names`` and ``default_host`` to the hostname clients will use (the same name on the Ingress). Include ``HelmChartBuilder`` in the project, then provision. Build and push the server image your chart references.

#. **Install the chart** — Enable cluster addons as needed (for example helm3, ingress, registry on MicroK8s). Install the Helm chart from the provision output.

#. **TLS Secret and Ingress** — Create a TLS Secret from your certificate and key, then an Ingress that references it and routes to the server Service.

   .. code-block:: bash

       kubectl create secret tls flare-tls -n default \
         --cert=path/to/your/tls.crt \
         --key=path/to/your/tls.key

   **Match your project’s ``scheme``**

   The Ingress configuration must align with the **transport scheme** you set in your project (or its default). Verify it in the provisioned server kit: ``startup/fed_server.json`` → ``servers[0].service.scheme`` (``grpc``, ``grpcs``, ``http``, or ``https``). Use the matching subsection below—annotations differ between gRPC and HTTP/WebSocket.

   .. note::
      **Ingress YAML always says ``http:``:** Kubernetes Ingress rules are declared under ``spec.rules[].http`` regardless of whether the backend is gRPC, WebSockets, or REST. That field is **API structure**, not a claim about the pod’s protocol. Use your project’s ``scheme`` (visible in ``fed_server.json``) to choose Ingress annotations, not the word ``http`` in the Ingress spec.

   **gRPC / ``grpcs``** — Traffic is gRPC over HTTP/2. When TLS terminates at the Ingress and the pod uses ``connection_security: clear``, the controller must speak **HTTP/2 (cleartext gRPC)** to the backend Service. For **ingress-nginx**, it is **correct** to set ``nginx.ingress.kubernetes.io/backend-protocol: "GRPC"`` in that case (``scheme`` ``grpc`` or ``grpcs`` only). Omit it for HTTP/WebSocket ``scheme``. Without it, gRPC-through-Ingress often fails. Other ingress controllers need their own gRPC or HTTP/2 backend settings.

   **HTTP / ``https`` (WebSocket driver)** — Many kits use ``scheme: http`` (or ``https`` with TLS on the server). The aio HTTP driver listens for **WebSocket** upgrades on path ``/f3`` (not gRPC). **Do not** set ``backend-protocol: GRPC`` in that case—it can break a working setup. For **ingress-nginx**, enable WebSocket proxying (for example ``nginx.ingress.kubernetes.io/websocket-services: "server"`` using your Service **name**), keep a path prefix that includes ``/f3`` (often ``path: /`` with ``pathType: Prefix`` is enough), and consider longer proxy read timeouts for long-lived connections. If your test already passes with a minimal HTTP Ingress, your scheme is likely HTTP/WebSocket and the gRPC annotation is unnecessary.

   **Single backend port** — The example below routes **one** Service port (for example ``8002``). It matches deployments where **federated learning and admin share the same listening port** (``fed_learn_port`` and ``admin_port`` are the same in the provisioned kit’s ``sp_end_point``, or your chart exposes a single port for both). If **``fed_learn_port`` and ``admin_port`` differ**, a single ``path: /`` rule to one port is not enough: use a **second** Ingress rule (different host or path) to the admin Service port, merge to one port in the project/Helm values, or use **TCP passthrough** (later section on this page) so the server terminates TLS on one TCP port for all traffic.

   Replace ``fl.example.com`` with your hostname. The backend port must match the server Service port in your project (example uses 8002). **Ingress manifest (gRPC scheme)** — include the ``GRPC`` annotation only when ``service.scheme`` is ``grpc`` or ``grpcs``:

   .. code-block:: yaml

       apiVersion: networking.k8s.io/v1
       kind: Ingress
       metadata:
         name: flare-ingress
         namespace: default
         annotations:
           nginx.ingress.kubernetes.io/ssl-passthrough: "false"
           nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
       spec:
         ingressClassName: nginx
         tls:
           - hosts:
               - fl.example.com
             secretName: flare-tls
         rules:
           - host: fl.example.com
             http:
               paths:
                 - path: /
                   pathType: Prefix
                   backend:
                     service:
                       name: server
                       port:
                         number: 8002

#. **Network and DNS** — Allow inbound HTTPS (port 443). Ensure the ingress hostname resolves for clients (DNS or ``/etc/hosts``).

***********************************************
TCP passthrough (server TLS/mTLS)
***********************************************

These steps assume MicroK8s with the **NGINX** ingress addon (Traefik uses different TCP configuration). Open inbound TCP 8002 on the node or security group.

Server deployment
=================

#. **Project** — Example below uses mTLS and a single port (8002). For TLS without client certificates, set ``connection_security`` to ``tls`` at project and server level. Replace ``YOUR_HOSTNAME`` with the hostname clients will use.

   .. code-block:: yaml

       api_version: 3
       name: k8s-flare-pattern-b
       description: TCP passthrough, server TLS/mTLS, single port 8002
       connection_security: mtls

       participants:
         - name: server
           type: server
           org: nvidia
           fed_learn_port: 8002
           connection_security: mtls
           host_names: [YOUR_HOSTNAME]
           default_host: YOUR_HOSTNAME

         - name: site-1
           type: client
           org: nvidia

         - name: site-2
           type: client
           org: nvidia

         - name: admin@nvidia.com
           type: admin
           org: nvidia
           role: project_admin

       builders:
         - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
           args:
             template_file:
               - master_template.yml
               - aws_template.yml
               - azure_template.yml
         - path: nvflare.lighter.impl.static_file.StaticFileBuilder
           args:
             config_folder: config
         - path: nvflare.lighter.impl.cert.CertBuilder
         - path: nvflare.lighter.impl.signature.SignatureBuilder
         - path: nvflare.lighter.impl.helm_chart.HelmChartBuilder
           args:
             docker_image: localhost:32000/nvfl-min:0.0.1

#. **Server image** — The pod only needs Python and NVFlare; configuration and certificates come from the mounted workspace at runtime.

   .. code-block:: dockerfile

       FROM python:3.11-slim
       RUN pip install --no-cache-dir -U pip \
           && pip install --no-cache-dir nvflare
       ENV PYTHONUNBUFFERED=1

   Save as ``Dockerfile``, build, and push to your registry (MicroK8s example):

   .. code-block:: bash

       microk8s enable registry
       # If push fails: add {"insecure-registries": ["localhost:32000"]} to /etc/docker/daemon.json, restart Docker.
       docker build -t localhost:32000/nvfl-min:0.0.1 -f Dockerfile .
       docker push localhost:32000/nvfl-min:0.0.1

#. **Provision** — Run provision locally or on any host with NVFlare, then copy the output directory (for example ``prod_00/``) to the machine that runs the cluster.

   .. code-block:: bash

       nvflare provision -p project.yml -w provision_workspace

#. **Helm install** — On the cluster host, install from the copied folder. ``workspace`` and ``persist`` should be absolute paths. The namespace here must match the TCP ConfigMap backend (for example ``default`` in ``default/server:8002``).

   .. code-block:: bash

       mkdir -p /tmp/nvflare
       cd /path/on/server/to/prod_00
       microk8s helm3 install \
         -n default \
         --set workspace="$(pwd)" \
         --set persist=/tmp/nvflare \
         nvflare-k8s-demo \
         nvflare_hc/

   Confirm the server pod is running and a Service named ``server`` exists in that namespace (the ingress TCP mapping depends on it):

   .. code-block:: bash

       microk8s kubectl get pods -n default
       microk8s kubectl get svc server -n default

#. **TCP ConfigMap** — MicroK8s NGINX reads mappings from ``nginx-ingress-tcp-microk8s-conf`` in namespace ``ingress``. Map external port ``8002`` to your server Service; the value must be a **quoted** string.

   .. code-block:: yaml

       apiVersion: v1
       kind: ConfigMap
       metadata:
         name: nginx-ingress-tcp-microk8s-conf
         namespace: ingress
       data:
         "8002": "default/server:8002"

   If the ConfigMap already exists, merge the key:

   .. code-block:: bash

       microk8s kubectl patch configmap nginx-ingress-tcp-microk8s-conf -n ingress --type merge \
         -p '{"data":{"8002":"default/server:8002"}}'

   Adjust ``namespace``, Service name, and port if yours differ.

#. **Ingress controller port** — Expose TCP 8002 on the NGINX ingress DaemonSet (the container that already has ports 80 and 443). Patch container index ``0`` first:

   .. code-block:: bash

       microk8s kubectl patch daemonset nginx-ingress-microk8s-controller -n ingress --type=json \
         -p='[{"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"name": "tcp-8002", "containerPort": 8002, "hostPort": 8002, "protocol": "TCP"}}]'

   If the patch fails, list container names and retry with ``containers/1`` (or the index that owns 80/443):

   .. code-block:: bash

       microk8s kubectl get ds nginx-ingress-microk8s-controller -n ingress -o jsonpath='{.spec.template.spec.containers[*].name}'

   .. code-block:: bash

       microk8s kubectl patch daemonset nginx-ingress-microk8s-controller -n ingress --type=json \
         -p='[{"op": "add", "path": "/spec/template/spec/containers/1/ports/-", "value": {"name": "tcp-8002", "containerPort": 8002, "hostPort": 8002, "protocol": "TCP"}}]'

   Restart the controller so pods pick up the change:

   .. code-block:: bash

       microk8s kubectl rollout restart daemonset nginx-ingress-microk8s-controller -n ingress

#. **Clients reach the server** — Open TCP 8002 to the node. Clients must resolve ``YOUR_HOSTNAME`` to that node (DNS or ``/etc/hosts``).

To confirm the deployment, check server logs (for example ``kubectl logs -l system=server -f -n default``), then connect a client or admin and run a job. If port 8002 still does not accept traffic, see :ref:`k8s_ingress_tls_troubleshooting`.

*********************************
Client deployment (mTLS)
*********************************

Use the same project as the server. Set ``connection_security: mtls`` on the project and server; set server ``host_names`` and ``default_host``. After a single provision run, each site has its own directory (for example ``site-1/``, ``site-2/``).

Outside the cluster
===================

Copy each site's directory to its host. Ensure the server address from that kit is reachable, then start the client:

.. code-block:: bash

    ./startup/start.sh

Inside the cluster
==================

Mount the client **site directory** (for example ``site-1/``) so ``startup/`` and ``local/`` appear under one path (for example ``/workspace``). **hostPath** must exist on the **node where the pod runs** (typical on single-node MicroK8s). Alternatively mount certs via a Secret and supply the rest of the kit from a volume; do not mix kits across sites.

**Run ``sub_start.sh`` in the foreground** — On a laptop, ``startup/start.sh`` backgrounds the real client and returns immediately; that is fine in an interactive shell. In Kubernetes the main process must stay running. Use ``sub_start.sh`` as PID 1:

.. code-block:: bash

    export PYTHONUNBUFFERED=1
    exec /workspace/startup/sub_start.sh

Mount the site root at ``/workspace`` so that path resolves. A minimal image is Python plus NVFlare (same idea as the server image).

Signed connection endpoint
--------------------------

Each client's ``fed_client.json`` contains ``overseer_agent.args.sp_end_point`` (``host:fed_learn_port:admin_port``). That file is signed: you cannot change the host or port without re-provisioning or using a new kit. During TLS, the client must connect to a hostname that appears on the **server certificate** (SAN). Those names come from the server participant's ``host_names`` and ``default_host`` at provision time.

Same cluster as the server
---------------------------

To reach the server via Kubernetes Service DNS (traffic stays in-cluster):

#. Add the Service hostname to the server participant's ``host_names`` (adjust namespace if the server is not in ``default``):

   .. code-block:: yaml

       host_names: [YOUR_PUBLIC_HOSTNAME, server.default.svc.cluster.local]
       default_host: YOUR_PUBLIC_HOSTNAME

#. For each client that runs in this cluster, set ``connect_to`` so only that site's kit uses the Service name. The ``name`` field must match the server participant's name in the project.

   .. code-block:: yaml

       - name: site-1
         type: client
         org: nvidia
         connect_to:
           name: server
           host: server.default.svc.cluster.local

   Clients without ``connect_to`` use ``default_host`` (typical for machines outside the cluster).

#. Provision, then deploy using **that site's** directory only. The in-cluster client's ``fed_client.json`` will point at ``server.default.svc.cluster.local``.

#. **Egress** — Default cluster networking allows this. If you use a default-deny NetworkPolicy, allow TCP from the client workload to the server Service on your FL port (for example 8002).

**Same cluster without reprovisioning** — If the kit already points at a hostname in ``sp_end_point`` (for example ``flaredevserv``) and that name is on the server certificate, add **hostAliases** so that hostname resolves to the **server Service ClusterIP** inside the pod. Traffic goes pod → Service → server; TLS still matches. Get the IP with ``kubectl get svc server -n <namespace> -o jsonpath='{.spec.clusterIP}'``. The alias hostname must **exactly** match the host part of ``sp_end_point``. If the Service is recreated, update the IP.

Another cluster or the public hostname
---------------------------------------

Kubernetes does not provide a cross-cluster ``*.svc.cluster.local`` name for your server. Clients elsewhere must use a hostname and port that actually route to the server (load balancer, node DNS, ingress, VPN DNS, and so on). Omit ``connect_to`` to use ``default_host``, or set ``connect_to.host`` per client when a site needs a different routable name than ``default_host``. Open cloud security groups, firewalls, and NetworkPolicy egress as needed. From a debug pod, ``nc -zv <host> <port>`` confirms reachability.

Checklist for a client pod
---------------------------

- **Endpoint:** Either reprovision with ``connect_to`` / ``host_names`` as above, or use **hostAliases** to map the kit hostname to the server ClusterIP.
- **Command:** ``sub_start.sh`` in the foreground, not ``start.sh``.
- **Mount:** Site root at a fixed path (for example ``/workspace``); ``hostPath`` only works on nodes that have that directory.
- **Image:** Python and NVFlare; ``PYTHONUNBUFFERED=1`` helps logs appear in ``kubectl logs``.

Outside the cluster, ``./startup/start.sh`` remains the usual entrypoint. Do not reuse one site's certificates for another.

.. _k8s_ingress_tls_troubleshooting:

***************************
Troubleshooting
***************************

**Ingress controller: "Error getting Service default/server"**

The Service ``server`` is missing in that namespace. Create it if the chart did not (selector must match server pod labels):

.. code-block:: yaml

    apiVersion: v1
    kind: Service
    metadata:
      name: server
      namespace: default
      labels:
        system: server
    spec:
      selector:
        system: server
      ports:
        - name: fl-port
          port: 8002
          targetPort: 8002
          protocol: TCP

Then restart the ingress controller:

.. code-block:: bash

    microk8s kubectl rollout restart daemonset nginx-ingress-microk8s-controller -n ingress

**Client pod exits immediately (Succeeded, blank logs)**

On-prem ``start.sh`` runs ``sub_start.sh`` in the background and exits, so the container finishes in about a second. Run ``sub_start.sh`` in the foreground instead (see *Inside the cluster* above).

**MicroK8s 1.35+ (Traefik)**

The default ingress addon may be Traefik. The ConfigMap and DaemonSet names in this page target NGINX. Configure TCP routing with Traefik's TCP or IngressRouteTCP resources instead.

**Signed startup files**

Do not edit ``fed_client.json`` or other signed files. Re-provision and redistribute kits when the server hostname or port must change.

**Moving the server to a new machine**

You can reuse the same kits: deploy the same server output on the new host and make the **original** hostname resolve to the new address (DNS or ``/etc/hosts`` on clients and admin). The existing server certificate still matches that hostname. Re-provision only when you want a new hostname in the certificates and in every kit.

**TLS at ingress: clients fail to connect**

Verify your project’s ``scheme`` (in ``fed_server.json`` → ``service.scheme``) matches the Ingress annotations. For **gRPC / grpcs**, confirm **ingress-nginx** has ``backend-protocol: "GRPC"`` when TLS ends at the Ingress and the pod is ``clear``. For **http / https**, ensure WebSockets to ``/f3`` are allowed and **remove** the gRPC annotation if you added it by mistake. If ``fed_learn_port`` and ``admin_port`` differ in the signed ``sp_end_point``, ensure both ports are reachable—two Ingress backends, one merged server port, or TCP passthrough instead of HTTP Ingress.
