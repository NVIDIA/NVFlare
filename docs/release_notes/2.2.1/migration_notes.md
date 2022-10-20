

# Migration to 2.2.1 notes and tips


### Using FOBS to serialize/deserialize data between Client and Server

* prior to NVFLARE 2.1.4, NVFLARE is using python pickle to transfer data between client and server, since 2.1.4, 
we switched to Flare Object Serializer (FOBS), you might experience failures if your code is still using Pickle. 
To migrate the code or you experience error due to this, please refer to [Flare Object Serializer (FOBS)](https://github.com/NVIDIA/NVFlare/tree/main/nvflare/fuel/utils/fobs/README.rst)

### Replace TLS certificates

* Since 2.2.1, we changed the authorization model from centralized to federated authorization. This implies you can not 
use the original startup kit (which contains the old TLS certificates). You will need to cleanup the old setartup kits
re-provision your project

### Use new Project.yml template

* Since 2.2.1, we also enabled federated site-policies, you will need to use new project.yml template. Please refer [default project.yml](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html#default-project-yml-file)  

