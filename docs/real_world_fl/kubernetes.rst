.. _eks_deployment:

############################################
Amazon Elastic Kubernetes Service Deployment
############################################
In this document, we will describe how to run the entire NVIDIA FLARE inside one Amazon Elastic Kubernetes Service (EKS).  For information
how to run NVIDIA FLARE inside microk8s (local kubernetes cluster), please refer to :ref:`_helm_chart`.  That document describes how to
provision one NVIDIA FLARE system, configure your microk8s cluster, deploy the servers, the overseer and the clients to that cluster, and
control and submit jobs to that NVIDIA FLARE from admin console.


Start the EKS
=============
We assume that you have one AWS account which allows you to start one EKS.  We also assume you have eksctl, aws and kubectl installed in your local machine.
Note that the versions of those CLI may affect the operations.  We suggest keep them updated.

The first thing is to start the EKS with eksctl.  The following is a sample yaml file, ``cluster.yaml``, to create EKS with one command.

.. code-block:: yaml

    apiVersion: eksctl.io/v1alpha5
    kind: ClusterConfig

    metadata:
    name: nvflare-cluster
    region: us-west-2
    tags:
        project: nvflare

    nodeGroups:
    - name: worker-node
        instanceType: t3.large
        desiredCapacity: 2

.. code-block:: shell

    eksctl create cluster -f cluster.yaml

After this, you will have one cluster with two `t3.large` EC2 nodes.


Provision
=========

With NVIDIA FLARE installed in your local machine, you can create one set of startup kits easily with ``nvflare provision``.  If there is a project.yml file
in your current working directory, ``nvflare provision`` will create a workspace directory.  If that project.yml file does not exist, ``nvflare provision`` will
create a sample project.yml for you.  For simplicity, we suggest you remove/rename any existing project.yml and workspace directory.  Then provision the
set of startup kits from scratch.  When selecting the sample project.yml during provisioning, select a non-HA one, as most clusters support HA easily.

After provisioning, you will have a workspace/example_project/prod_00 folder, which includes server, site-1, site-2 and admin@nvidia.com folders.  If you
would like to use other names instead of ``site-1``, ``site-2``, etc, you can remove the workspace folder and modify the project.yml file.  After that,
you can run ``nvflare provision`` command to get the new set of startup kits.

Persistent Volume
=================

EKS provides several ways to create persistent volumes.  Before you can create the volume,
you need to create an OIDC provider, add a service account, and attach a policy to two roles: the node instance group and the service account.

.. code-block:: shell

    eksctl utils associate-iam-oidc-provider --region=us-west-2 --cluster=nvflare-cluster --approve

.. code-block:: shell

    eksctl create iamserviceaccount \
    --region us-west-2 \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster nvflare-cluster \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --approve \
    --role-only \
    --role-name AmazonEKS_EBS_CSI_DriverRole


.. code-block:: shell
    
    eksctl create addon --name aws-ebs-csi-driver \
    --cluster nvflare-cluster \
    --service-account-role-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/AmazonEKS_EBS_CSI_DriverRole \
    --force
    
The following is the policy json file that you have to attach to the roles.

.. code-block:: json

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "Poicly4EKS",
                "Effect": "Allow",
                "Action": [
                    "ec2:DetachVolume",
                    "ec2:AttachVolume",
                    "ec2:DeleteVolume",
                    "ec2:DescribeInstances",
                    "ec2:DescribeTags",
                    "ec2:DeleteTags",
                    "ec2:CreateTags",
                    "ec2:DescribeVolumes",
                    "ec2:CreateVolume"
                ],
                "Resource": [
                    "*"
                ]
            }
        ]
    }

The following yaml file will utilize EKS gp2 StorageClass to allocate 5GiByte space.  You
can run ``kubectl apply -f volume.yaml`` to make the volume available.

.. code-block:: yaml

    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
        name: nvflare-pv-claim
        labels:
            app: nvflare 
    spec:
        accessModes:
            - ReadWriteOnce
        resources:
            requests:
                storage: 5Gi
        storageClassName: gp2

After that, your EKS persistent volume should be waiting for the first claim.


Start Helper Pod
================

Now you will need to copy your startup kits to your EKS cluster.  Those startup kits will be copied into the volume you just created.
In order to access the volume, we deploy a helper pod which mounts that persistent volume and use kubectl cp to copy files from your
local machine to the cluster.

The following is the helper pod yaml file.

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    labels:
        run: bb8
    name: bb8
    spec:
    replicas: 1
    selector:
        matchLabels:
        run: bb8
    template:
        metadata:
        labels:
            run: bb8
        spec:
        containers:
        - args:
            - sleep
            - "50000"
            image: busybox
            name: bb8
            volumeMounts:
            - name: nvfl
                mountPath: /workspace/nvfl/
        volumes:
            - name: nvfl
            persistentVolumeClaim:
                claimName: nvflare-pv-claim


All pods can be deployed with ``kubectl apply -f`` so we just need the following command.

.. code-block:: shell

    kubectl apply -f bb8.yaml

Your helper pod should be up and running very soon.  Now copy the startup kits to the cluster with

.. code-block:: shell

    kubectl cp workspace/example_project/prod_00/server <helper-pod>:/workspace/nvfl/

And the same for site-1, site-2, admin@nvidia.com.

This will make the entire startup kits available at the nvflare-pv-claim of the cluster so that NVIDIA FLARE system
can mount that nvflare-pv-claim and access the startup kits.

After copying those folders to nvflare-pv-claim, you can shutdown the helper pod. The nvflare-pv-claim and its contents will remain available to 
server, client, and admin pods.

Start Server Pod
================

The NVIDIA FLARE server consists of two portions for Kubernetes clusters.  As you might know, 
the server needs computation to handle model updates, aggregations and other operations.  It also needs to provide a service for clients and admins
to connect.  Therefore, the followings are two separate yaml files that work together to create the NVIDIA FLARE server in EKS.

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    labels:
        run: nvflare
    name: nvflare 
    spec:
    replicas: 1
    selector:
        matchLabels:
        run: nvflare
    template:
        metadata:
        labels:
            run: nvflare
        spec:
        containers:
        - args:
            - -u
            - -m
            - nvflare.private.fed.app.server.server_train
            - -m
            - /workspace/nvfl/server
            - -s
            - fed_server.json
            - --set
            - secure_train=true
            - config_folder=config
            - org=nvidia
            command:
            - /usr/local/bin/python3
            image: nvflare/nvflare:2.4.0
            imagePullPolicy: Always
            name: nvflare
            volumeMounts:
            - name: nvfl
                mountPath: /workspace/nvfl/
        volumes:
            - name: nvfl
            persistentVolumeClaim:
                claimName: nvflare-pv-claim


.. code-block:: yaml
    
    apiVersion: v1
    kind: Service
    metadata:
    labels:
        run: server
    name: server
    spec:
    ports:
    - port: 8002
        protocol: TCP
        targetPort: 8002
        name: flport
    - port: 8003
        protocol: TCP
        targetPort: 8003
        name: adminport
    selector:
        run: nvflare

    
Note that the pod will use nvflare/nvflare:2.4.0 container image from dockerhub.com.  This image only includes the necessary dependencies to start
NVIDIA FLARE system.  If you require additional dependencies, such as Torch or MONAI, you will need to build and publish your own image and update
the yaml file accordingly.

Start Client Pods
=================
    
For the client pods, we only need one yaml file for eacch client.  The following is the deployment yaml file for site-1.

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    labels:
        run: site1
    name: site1
    spec:
    replicas: 1
    selector:
        matchLabels:
        run: site1
    template:
        metadata:
        labels:
            run: site1
        spec:
        containers:
        - args:
            - -u
            - -m
            - nvflare.private.fed.app.client.client_train
            - -m
            - /workspace/nvfl/site-1
            - -s
            - fed_client.json
            - --set
            - secure_train=true
            - uid=site-1
            - config_folder=config
            - org=nvidia
            command:
            - /usr/local/bin/python3
            image: nvflare/nvflare:2.4.0
            imagePullPolicy: Always
            name: site1
            volumeMounts:
            - name: nvfl
                mountPath: /workspace/nvfl/
        volumes:
            - name: nvfl
            persistentVolumeClaim:
                claimName: nvflare-pv-claim

Once the client is up and running, you can check the server log with ``kubectl logs`` and the log should show the clients registered.

Start and Connect to Admin Pods
===============================

We can also run the admin console inside the EKS cluster to submit jobs to the NVIDIA FLARE running in the EKS cluster.  Start the admin pod
with the following yaml file.

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    labels:
        run: admin
    name: admin
    spec:
    replicas: 1
    selector:
        matchLabels:
        run: admin
    template:
        metadata:
        labels:
            run: admin
        spec:
        containers:
        - args:
            - "50000" 
            command:
            - /usr/bin/sleep
            image: nvflare/nvflare:2.4.0
            imagePullPolicy: Always
            name: admin
            volumeMounts:
            - name: nvfl
                mountPath: /workspace/nvfl/
        volumes:
            - name: nvfl
            persistentVolumeClaim:
                claimName: nvflare-pv-claim

Once the admin pod is running, you can enter the pod with ``kubectl exec`` , cd to ``/workspace/nvfl/admin@nvidia.com/startup`` and run ``fl_admin.sh``.


Note that you need to copy the job from your local machine to the EKS cluster so that the ``transfer`` directory of admin@nvidia.com contains the jobs
you would like to run in that EKS cluster.

