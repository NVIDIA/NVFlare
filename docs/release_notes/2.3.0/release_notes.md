
### Release 2.3.0

* **Cloud Deployment Support**

  We expand the user support for cloud deployments in both Azure and AWS. Users can use simple CLI command to create infrastructures,
  deploy and start dashboard, FL server, FL Client(s).
```  
   nvflare dashboard --cloud azure | aws
   nvflare server --cloud azure | aws
   nvflare client --cloud azure | aws
```

* **Python Version Support**

  Flare expand the python version support and dropped the Python 3.7 support, FLARE 2.3.0 will support 3.8, 3.9, 3.10


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
    * scikit-learn linear model ( linea and logistics regression )
    * scikit-learn SVM
    * scikit-learn K-Means
    * XGBoost Random Forest


* **Vertical Learning**

    * **Federated Private Set Intersection (PSI)**
      In order to support vertical learning use cases such as secure user-id matching and feature
      over-lapping discovery, we have developed a multi-party private set intersection (PSI) operator
      that allows for the secure discovery of data intersections. Our approach leverages OpenMined's two-party
      [Private Set Intersection Cardinality protocol](https://github.com/OpenMined/PSI), which is based on ECDH and Bloom Filters, and we have
      made this protocol available for multi-party use. More information on our approach and how to use the
      PSI operator can be found in the [PSI Example](https://github.com/NVIDIA/NVFlare/blob/dev/examples/advanced/psi/README.md).
      It is worth noting that PSI is used as a pre-training step in the split learning example, which can be found in this
      [notebook](https://github.com/NVIDIA/NVFlare/blob/dev/examples/tutorial/vertical_federated_learning/cifar10-splitnn/cifar10_split_learning.ipynb).


* **Split-Learning**


* **Research Area**

    * FedSM
    * Data privacy risk detection tool


* **FLARE Communication**

  We laid some ground work in FLARE communication layer.
  The new communication features will be greatly improve the support protocol,
  performance and capabilities. We will make it General available in next release.


### Migrations tips
* [migrate to 2.2.1](docs/release_notes/2.2.1/migration_notes.md).
* [migrate to 2.3.0](docs/release_notes/2.3.0/migration_notes.md).

 