
### Release 2.3.0

* **Cloud Deployment Support**

  We expand the user support for cloud deployments in both Azure and AWS. Users can use simple CLI command to create infrastructures,
  deploy and start dashboard, FL server, FL Client(s).
```  
   nvflare dashboard --cloud azure | aws
   <server-startup-kit>/start.sh --cloud azure | aws
   <client-startup-kit>/start.sh --cloud azure | aws
```

* **Python Version Support**

  FLARE expands the python version support and dropped the Python 3.7 support, FLARE 2.3.0 will support 3.8, 3.9, 3.10


* **Flare API**

  We introduce an improved version of Admin API to make it easier to use in notebook env. The current admin APIs still work.
  we potentially could deprecate the old Admin API in the future. FLARE API currently support selected commands of admin APIs
  For details of the Flare API, you can check [this notebook](https://github.com/NVIDIA/NVFlare/blob/dev/examples/tutorial/flare_api.ipynb).
  If you consider to migrating the existing Admin API to this new API, there is the [migration guide](https://nvflare.readthedocs.io/en/dev/real_world_fl/migrating_to_flare_api.html)


* **Sign Custom Code**

  Before a job is submitted to the server, the submitter's private key is used to sign
  each file's digest.  Each folder has one signature file, which maps file names to signatures
  of all files inside that folder.  The verification is performed at deployment time.


* **Client-Side Model Initialization**

  Prior to FLARE 2.3.0, the model initialization is performed on the server-side.
  The model is either initialized via model file or custom model initiation code. Pre-defining a model file means to pre-generate and save the model file and then send over to the server.  
  Many users choose to run a custom model initialization code on server. But this, to some customers, could be a security risk.

  FLARE 2.3.0, introduce another way to initialize model on the client side, the Flare server will either can select
  the initial model based on user-choose strategy. Here is the [example](https://github.com/NVIDIA/NVFlare/tree/dev/examples/hello-world/hello-pt) using client-side model.
  You can read more about this feature in [FLARE documentation](TODO)


* **Tradition Machine Learning Examples**

  We add several examples to support federated learning using traditional machine learning algorithms.
  In particular,
    * [Linear model](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/sklearn-linear) using scikit-learn library via [iterative SGD training](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html). Linear and logistic regressions can be implemented following this iterative example by adopting different loss functions.
    * [SVM](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/sklearn-svm) using scikit-learn library. In this two-step process, server performs an additional round of SVM over the collected supporting vectors from clients.
    * [K-Means](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/sklearn-kmeans) using scikit-learn library via [mini-batch K-Means method](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html). In this iterative process, each client performs mini-batch K-Means and server syncs the updates for global model. 
    * [Random Forest](https://github.com/NVIDIA/NVFlare/tree/dev/examples/advanced/random_forest) using XGBoost library with [random forest functionality](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html). In this two-step process, clients construct sub-forests on their local data, and server ensembles all collected sub-forests to produce the global random forest. 


* **Vertical Learning**

    * **Federated Private Set Intersection (PSI)**
      In order to support vertical learning use cases such as secure user-id matching and feature
      over-lapping discovery, we have developed a multi-party private set intersection (PSI) operator
      that allows for the secure discovery of data intersections. Our approach leverages OpenMined's two-party
      [Private Set Intersection Cardinality protocol](https://github.com/OpenMined/PSI), which is based on ECDH and Bloom Filters, and we have
      made this protocol available for multi-party use. More information on our approach and how to use the
      PSI operator can be found in the [PSI Example](https://github.com/NVIDIA/NVFlare/blob/dev/examples/advanced/psi/README.md).
      It is worth noting that PSI is used as a pre-processing step in the split learning example, which can be found in this
      [notebook](https://github.com/NVIDIA/NVFlare/blob/dev/examples/advanced/vertical_federated_learning/cifar10-splitnn/README.md).
      
    * **Split-Learning** can allow the training of deep neural networks on vertically separated data. 
      With this release, we include an [example](https://github.com/NVIDIA/NVFlare/blob/dev/examples/advanced/vertical_federated_learning/cifar10-splitnn/README.md) 
      on how to run [split learning](https://arxiv.org/abs/1810.06060) using the CIFAR-10 dataset assuming one client holds the images, 
      and the other client holds the labels to compute losses and accuracy metrics. 
      Activations and corresponding gradients are being exchanged between the clients using FLARE's new communication API.


* **Research Area**

    * **FedSM** This example illustrates the personalized federated learning algorithm [FedSM](https://arxiv.org/abs/2203.10144) accepted to CVPR2022. It bridges the different data distributions across clients via a SoftPull mechanism and utilizes a Super Model. A model selector is trained to predict the belongings of a particular sample to any of the clients' personalized models or global model. The training of this model also illustrates a challenging federated learning scenario with extreme label-imbalance, where each local training is only based on a single label towards the optimization for classification of a number of classes equvilant to the number of clients. In this case, the higher-order moments of the Adam optimizer are also averaged and synced together with model updates. 
    * Data privacy risk detection tool


* **FLARE Communication**

  We laid some ground work in FLARE communication layer.
  The new communication features will be greatly improve the support protocol,
  performance and capabilities. We will make it General available in next release.


### Migrations tips
* [migrate to 2.2.1](docs/release_notes/2.2.1/migration_notes.md).
* [migrate to 2.3.0](docs/release_notes/2.3.0/migration_notes.md).

 
