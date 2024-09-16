####################################################
Integration of Flower Applications with NVIDIA FLARE
####################################################

`Flower <https://flower.ai>`_ is an open-source project that implements a unified approach
to federated learning, analytics, and evaluation. Flower has developed a large set of
strategies and algorithms for FL application development and a healthy FL research community. 

FLARE, on the other hand, has been focusing on providing an enterprise-ready, robust runtime
environment for FL applications. 

With the integration of Flower and FLARE, applications developed with the Flower framework
will run easily in FLARE runtime without needing to make any changes. All the user needs to do
is configure the Flower application into a FLARE job and submit the job to the FLARE system.


.. toctree::
   :maxdepth: 1

   flower_integration/flower_initial_integration
   flower_integration/flower_job_structure
   flower_integration/flower_run_as_flare_job
   flower_integration/flare_multi_job_architecture
   flower_integration/flower_detailed_design
   flower_integration/flower_reliable_messaging
