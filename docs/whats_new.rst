.. _whats_new:

##########
What's New
##########

What's New in FLARE v2.3.0
==========================

FL Split Learning
-----------------
New example with FL Split Learning.

Cloud User Experience
----------------------
The Dashboard UI now generates scripts for launching the FL Server and Clients in the cloud as well as running the Dashboard itself
in the cloud.

Communicaton Framework Upgrades
-------------------------------
There should be no visible changes in terms of the configuration and uage patterns for the end user, but the underlying communication
layer has been improved to allow for greater flexibility and performance.

FLARE API to provide better user experience than FLAdminAPI
-----------------------------------------------------------
See :ref:`migrating_to_flare_api` for details on migrating to the new FLARE API. For now, the FLAdminAPI should still remain
functional.

Job Signing for Improved Security
---------------------------------
Before a job is submitted to the server, the submitter's private key is used to sign each file's digest.  Each folder has one signature
file, which maps file names to the signatures of all files inside that folder. The signer's certificate is also included for signature verification.
The verification is performed at deployment time, rather than submission time, as the clients do not receive the job until the job is deployed.

Support for Python 3.9 and Python 3.10
--------------------------------------
FLARE is now supported for Python 3.9 and Python 3.10. Note that Python 3.7 is no longer actively supported and tested.


Previous Releases of FLARE
==========================

.. toctree::
   :maxdepth: 1

   release_notes/flare_221
   release_notes/flare_220
   release_notes/flare_210
