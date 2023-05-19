.. _helm_chart:

###########################
Helm Chart for NVIDIA FLARE
###########################

Sometimes, users would like to deploy NVIDIA FLARE to an existing Kubernetes cluster.  Now
the provisioning tool includes a new builder ``HelmChartBuilder`` that can generate a reference
Helm Chart for users to deploy NVIDIA FLARE to a local microk8s Kubernetes instance.

.. note::

    The generated Helm Chart is a starting point and serves as a reference.  Depending on the Kubernetes cluster,
    users may need to modify and/or to perform additional operations to successfully deploy the chart.
    

.. note::

    The following document assumes users have microk8s (common bundle in ubuntu server 20.04 and above) running on his local machine.
    With the helm chart, users are able to start the overseer, servers in the k8s cluster after provisioning.
    The clients and admin console can connect to the overseer and servers in the k8s cluster.


***************************
Update on provisioning tool
***************************

In order to generate the helm chart, add the HelmChartBuilder to the project.yml file.

.. code-block:: yaml

    - path: nvflare.lighter.impl.helm_chart.HelmChartBuilder
        args:
        docker_image: localhost:32000/nvfl-min:0.0.1


The ``docker_image`` is the actual image used for all pods running in the k8s.  The provisioners have 
to build it separately and make sure it is available to the k8s cluster.  For microk8s, enabling the docker registry 
server by running this:

.. code-block:: shell

    microk8s enable registry

This will create a registry server listening to port 32000

********************
Provisioning results
********************

Running provision command as usual, either in the new format ``nvflare provision`` or just ``provision``.

After the command, there should a folder with structure similar to the following:

.. code-block:: shell

    $ tree -L 1
    .
    ├── admin@nvidia.com
    ├── compose.yaml
    ├── nvflare_compose
    ├── nvflare_hc
    ├── overseer
    ├── server1
    ├── server2
    ├── site-1
    └── site-2

    8 directories, 1 file

Note there is a nvflare_hc folder.  This folder is the Helm Chart package.


******************
Preparing microk8s
******************

Enabling microk8s addons
========================
NVIDIA FLARE Helm Chart depends on a few services (aka addons in microk8s) provided by the Kubernetes cluster.  Please
check if they are enabled.

.. code-block:: shell

    $ microk8s status
    microk8s is running
    high-availability: no
    datastore master nodes: 127.0.0.1:19001
    datastore standby nodes: none
    addons:
    enabled:
        dns                  # (core) CoreDNS
        ha-cluster           # (core) Configure high availability on the current node
        helm3                # (core) Helm 3 - Kubernetes package manager
        hostpath-storage     # (core) Storage class; allocates storage from host directory
        ingress              # (core) Ingress controller for external access
        registry             # (core) Private image registry exposed on localhost:32000
        storage              # (core) Alias to hostpath-storage add-on, deprecated
    disabled:
        community            # (core) The community addons repository
        dashboard            # (core) The Kubernetes dashboard
        gpu                  # (core) Automatic enablement of Nvidia CUDA
        helm                 # (core) Helm 2 - the package manager for Kubernetes
        host-access          # (core) Allow Pods connecting to Host services smoothly
        mayastor             # (core) OpenEBS MayaStor
        metallb              # (core) Loadbalancer for your Kubernetes cluster
        metrics-server       # (core) K8s Metrics Server for API access to service metrics
        prometheus           # (core) Prometheus operator for monitoring and logging
        rbac                 # (core) Role-Based Access Control for authorisation

If any of the enabled services are not enabled in your environment, please enable it.  The following example shows how
to enable helm3 addon.

.. code-block:: shell

    $ microk8s enable helm3
    Infer repository core for addon helm3
    Enabling Helm 3
    Fetching helm version v3.8.0.
    % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                    Dload  Upload   Total   Spent    Left  Speed
    100 12.9M  100 12.9M    0     0  11.5M      0  0:00:01  0:00:01 --:--:-- 11.5M
    Helm 3 is enabled


Allowing network traffic
========================

We have to change the cluster to allow incoming network traffic, such as those
from admin consoles and NVIDIA FLARE clients, to enter the cluster.  After the network
traffic enters the cluster, the cluster also needs to know how to route the traffice
to the deployed services.


Users have to enable ingress controller and modify some configuration of microk8s cluster.

Complete the following steps to enable microk8s to open and route
network traffic to overseer and servers.


Edit configmap of ingress to route traffic
------------------------------------------

.. code-block:: shell

    $ microk8s kubectl edit cm nginx-ingress-tcp-microk8s-conf -n ingress

Add this section to the configmap

.. code-block:: yaml

    data:
        "8002": default/server1:8002
        "8003": default/server1:8003
        "8102": default/server2:8102
        "8103": default/server2:8103
        "8443": default/overseer:8443

Edit DaemonSet of ingress to open ports
---------------------------------------

.. code-block:: shell

    $ microk8s kubectl edit ds nginx-ingress-microk8s-controller -n ingress

Add this section at (spec.template.spec.containers[0].ports)

.. code-block:: yaml

        - containerPort: 8443
          hostPort: 8443
          name: overseer
          protocol: TCP
        - containerPort: 8002
          hostPort: 8002
          name: server1fl
          protocol: TCP
        - containerPort: 8003
          hostPort: 8003
          name: server1adm
          protocol: TCP
        - containerPort: 8102
          hostPort: 8102
          name: server2fl
          protocol: TCP
        - containerPort: 8103
          hostPort: 8103
          name: server2adm
          protocol: TCP


*********************
Installing helm chart
*********************

To install the helm chart, with microk8s environment, run the following command in the same directory as previous section.

.. code-block:: shell

    $ mkdir -p /tmp/nvflare
    $ microk8s helm3 install --set workspace=$(pwd) --set svc-persist=/tmp/nvflare nvflare-helm-chart-demo nvflare_hc/

    NAME: nvflare-helm-chart-demo
    LAST DEPLOYED: Fri Sep 23 12:28:24 2022
    NAMESPACE: default
    STATUS: deployed
    REVISION: 1
    TEST SUITE: None

Here the ``nvflare-helm-chart-demo`` is the name we choose for this installed application.  You can choose a different name so
that it's easy to recognize the deployed application.

The ``nvflare_hc/`` is the folder provisioning tool generated, as shown in the previous section.  You can take a look at files in
that folder and feel free to change them for your own environment.

.. note::

    Here we use the host's /tmp/nvflare as the persist storage space for all pods in microk8s.  Please make sure
    that directory exists before running the above command
    
****************************************
Verifying NVIDIA FLARE is up and running
****************************************

You can use ``kubectl`` to check the status of NVIDIA FLARE application, installed by the chart.  For example, in
microk8s environment, run the following command to see if overseer and servers are started.

.. code-block:: shell

    $ microk8s kubectl get pods
    NAME                        READY   STATUS    RESTARTS       AGE
    dnsutils                    1/1     Running   74 (13m ago)   62d
    server1-7675668544-xvfvp    1/1     Running   0              4m50s
    overseer-6f9dd66c97-n7bkd   1/1     Running   0              4m50s
    server2-86bc4fc87f-s9n2s    1/1     Running   0              4m50s

The ``dnsutils`` is a built-in addon for dns service inside microk8s.  You can ignore it.

For more details on the pods inside Kubernetes cluster, you can run the following command.

.. code-block:: shell

    $ microk8s kubectl describe pods
    Name:         dnsutils
    Namespace:    default
    Priority:     0
    Node:         demolaptop/192.168.1.96
    Start Time:   Fri, 22 Jul 2022 13:36:54 -0700
    Labels:       <none>
    Annotations:  cni.projectcalico.org/containerID: 9cfa2cfbb4ef7b11b10c5793965e2a42682dea5d0b05b4454b4232da9ded6a8e
                cni.projectcalico.org/podIP: 10.1.179.67/32
                cni.projectcalico.org/podIPs: 10.1.179.67/32
    Status:       Running
    IP:           10.1.179.67
    IPs:
    IP:  10.1.179.67
    Containers:
    dnsutils:
        Container ID:  containerd://3c31a42f9c5dc10452d2af0a503682cd78e25a4b078877f96a1174d1156a23a5
        Image:         k8s.gcr.io/e2e-test-images/jessie-dnsutils:1.3
        Image ID:      k8s.gcr.io/e2e-test-images/jessie-dnsutils@sha256:8b03e4185ecd305bc9b410faac15d486a3b1ef1946196d429245cdd3c7b152eb
        Port:          <none>
        Host Port:     <none>
        Command:
        sleep
        3600
        State:          Running
        Started:      Fri, 23 Sep 2022 12:19:55 -0700
        Last State:     Terminated
        Reason:       Unknown
        Exit Code:    255
        Started:      Thu, 18 Aug 2022 11:18:34 -0700
        Finished:     Fri, 23 Sep 2022 12:19:25 -0700
        Ready:          True
        Restart Count:  74
        Environment:    <none>
        Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-f4sxs (ro)
    Conditions:
    Type              Status
    Initialized       True 
    Ready             True 
    ContainersReady   True 
    PodScheduled      True 
    Volumes:
    kube-api-access-f4sxs:
        Type:                    Projected (a volume that contains injected data from multiple sources)
        TokenExpirationSeconds:  3607
        ConfigMapName:           kube-root-ca.crt
        ConfigMapOptional:       <nil>
        DownwardAPI:             true
    QoS Class:                   BestEffort
    Node-Selectors:              <none>
    Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                                node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
    Events:                      <none>


    Name:         server1-7675668544-xvfvp
    Namespace:    default
    Priority:     0
    Node:         demolaptop/192.168.1.96
    Start Time:   Fri, 23 Sep 2022 12:28:25 -0700
    Labels:       pod-template-hash=7675668544
                system=server1
    Annotations:  cni.projectcalico.org/containerID: 7493a356143ad0c4e4fdbe781d995c01d52c4caa31e961066d4a8769dfa1d360
                cni.projectcalico.org/podIP: 10.1.179.94/32
                cni.projectcalico.org/podIPs: 10.1.179.94/32
    Status:       Running
    IP:           10.1.179.94
    IPs:
    IP:           10.1.179.94
    Controlled By:  ReplicaSet/server1-7675668544
    Containers:
    server1:
        Container ID:  containerd://16928775549dbf9cb2d68eea6412e682a170f72b5dbcdbf8c56790c8b9a30fd5
        Image:         localhost:32000/nvfl-min:0.0.1
        Image ID:      localhost:32000/nvfl-min@sha256:71658dc82b15e6cd5a2580c78e56011d166a70e1ff098306c93584c82cb63821
        Ports:         8002/TCP, 8003/TCP
        Host Ports:    0/TCP, 0/TCP
        Command:
        /usr/local/bin/python3
        Args:
        -u
        -m
        nvflare.private.fed.app.server.server_train
        -m
        /workspace/server1
        -s
        fed_server.json
        --set
        secure_train=true
        config_folder=config
        State:          Running
        Started:      Fri, 23 Sep 2022 12:28:27 -0700
        Ready:          True
        Restart Count:  0
        Environment:    <none>
        Mounts:
        /tmp/nvflare from svc-persist (rw)
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-hkhhq (ro)
        /workspace from workspace (rw)
    Conditions:
    Type              Status
    Initialized       True 
    Ready             True 
    ContainersReady   True 
    PodScheduled      True 
    Volumes:
    workspace:
        Type:          HostPath (bare host directory volume)
        Path:          /home/nvflare/workspace/nvf_hc_test/demo
        HostPathType:  Directory
    svc-persist:
        Type:          HostPath (bare host directory volume)
        Path:          /tmp/nvflare
        HostPathType:  Directory
    kube-api-access-hkhhq:
        Type:                    Projected (a volume that contains injected data from multiple sources)
        TokenExpirationSeconds:  3607
        ConfigMapName:           kube-root-ca.crt
        ConfigMapOptional:       <nil>
        DownwardAPI:             true
    QoS Class:                   BestEffort
    Node-Selectors:              <none>
    Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                                node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
    Events:                      <none>


    Name:         overseer-6f9dd66c97-n7bkd
    Namespace:    default
    Priority:     0
    Node:         demolaptop/192.168.1.96
    Start Time:   Fri, 23 Sep 2022 12:28:25 -0700
    Labels:       pod-template-hash=6f9dd66c97
                system=overseer
    Annotations:  cni.projectcalico.org/containerID: e9f6f2efb548c16217377eaaa8b79534a67e016277c3a0933d202d04904f46dc
                cni.projectcalico.org/podIP: 10.1.179.80/32
                cni.projectcalico.org/podIPs: 10.1.179.80/32
    Status:       Running
    IP:           10.1.179.80
    IPs:
    IP:           10.1.179.80
    Controlled By:  ReplicaSet/overseer-6f9dd66c97
    Containers:
    overseer:
        Container ID:  containerd://82426e5e414b863fff1cc4c8963a3e18acd49ff1ccb51befaf5c984f3ad0f1a4
        Image:         localhost:32000/nvfl-min:0.0.1
        Image ID:      localhost:32000/nvfl-min@sha256:71658dc82b15e6cd5a2580c78e56011d166a70e1ff098306c93584c82cb63821
        Port:          8443/TCP
        Host Port:     0/TCP
        Command:
        /workspace/overseer/startup/start.sh
        State:          Running
        Started:      Fri, 23 Sep 2022 12:28:27 -0700
        Ready:          True
        Restart Count:  0
        Environment:    <none>
        Mounts:
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-dz7qz (ro)
        /workspace from workspace (rw)
    Conditions:
    Type              Status
    Initialized       True 
    Ready             True 
    ContainersReady   True 
    PodScheduled      True 
    Volumes:
    workspace:
        Type:          HostPath (bare host directory volume)
        Path:          /home/nvflare/workspace/nvf_hc_test/demo
        HostPathType:  Directory
    kube-api-access-dz7qz:
        Type:                    Projected (a volume that contains injected data from multiple sources)
        TokenExpirationSeconds:  3607
        ConfigMapName:           kube-root-ca.crt
        ConfigMapOptional:       <nil>
        DownwardAPI:             true
    QoS Class:                   BestEffort
    Node-Selectors:              <none>
    Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                                node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
    Events:                      <none>


    Name:         server2-86bc4fc87f-s9n2s
    Namespace:    default
    Priority:     0
    Node:         demolaptop/192.168.1.96
    Start Time:   Fri, 23 Sep 2022 12:28:25 -0700
    Labels:       pod-template-hash=86bc4fc87f
                system=server2
    Annotations:  cni.projectcalico.org/containerID: 8ac76a0bfad2e4f0b1de9115f0d46c1a0dbacabb847c6160b1f144e82720fe99
                cni.projectcalico.org/podIP: 10.1.179.96/32
                cni.projectcalico.org/podIPs: 10.1.179.96/32
    Status:       Running
    IP:           10.1.179.96
    IPs:
    IP:           10.1.179.96
    Controlled By:  ReplicaSet/server2-86bc4fc87f
    Containers:
    server2:
        Container ID:  containerd://c1e530fc6fc320d9b9388d81727440324cc11e0bb61e3b3e76a2362638f89357
        Image:         localhost:32000/nvfl-min:0.0.1
        Image ID:      localhost:32000/nvfl-min@sha256:71658dc82b15e6cd5a2580c78e56011d166a70e1ff098306c93584c82cb63821
        Ports:         8102/TCP, 8103/TCP
        Host Ports:    0/TCP, 0/TCP
        Command:
        /usr/local/bin/python3
        Args:
        -u
        -m
        nvflare.private.fed.app.server.server_train
        -m
        /workspace/server2
        -s
        fed_server.json
        --set
        secure_train=true
        config_folder=config
        State:          Running
        Started:      Fri, 23 Sep 2022 12:28:28 -0700
        Ready:          True
        Restart Count:  0
        Environment:    <none>
        Mounts:
        /tmp/nvflare from svc-persist (rw)
        /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-6cwbh (ro)
        /workspace from workspace (rw)
    Conditions:
    Type              Status
    Initialized       True 
    Ready             True 
    ContainersReady   True 
    PodScheduled      True 
    Volumes:
    workspace:
        Type:          HostPath (bare host directory volume)
        Path:          /home/nvflare/workspace/nvf_hc_test/demo
        HostPathType:  Directory
    svc-persist:
        Type:          HostPath (bare host directory volume)
        Path:          /tmp/nvflare
        HostPathType:  Directory
    kube-api-access-6cwbh:
        Type:                    Projected (a volume that contains injected data from multiple sources)
        TokenExpirationSeconds:  3607
        ConfigMapName:           kube-root-ca.crt
        ConfigMapOptional:       <nil>
        DownwardAPI:             true
    QoS Class:                   BestEffort
    Node-Selectors:              <none>
    Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                                node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
    Events:                      <none>



************************
Login with admin console
************************

Now on another terminal, with nvflare installed and /etc/hosts modified to 
include the IP of overseer, server1 and server2, which is the IP of the 
machine running the microk8s cluster, run fl_admin.sh of admin@nvidia.com/startup.  
Login as admin@nvidia.com.

For example: /etc/hosts is modified as (if microk8s is running at 192.168.1.123 and clients and admin console is running at slowdesktop machine)

.. code-block:: shell

    $ cat /etc/hosts
    127.0.0.1       localhost
    127.0.1.1       slowdesktop
    192.168.1.123 overseer server1 server2
    # The following lines are desirable for IPv6 capable hosts
    ::1     ip6-localhost ip6-loopback
    fe00::0 ip6-localnet
    ff00::0 ip6-mcastprefix
    ff02::1 ip6-allnodes
    ff02::2 ip6-allrouters


***********************
Uninstalling helm chart
***********************

Users can uninstall the chart by running (note ``nvflare-helm-chart-demo`` is the release name we used when installing the chart)

.. code-block:: shell
    
    $ microk8s helm3 uninstall nvflare-helm-chart-demo

