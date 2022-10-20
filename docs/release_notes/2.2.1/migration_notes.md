

# Migration to 2.2.1 notes and tips


### Using FOBS to serialize/deserialize data between Client and Server

* prior to NVFLARE 2.1.4, NVFLARE is using python pickle to transfer data between client and server, since 2.1.4, 
we switched to Flare Object Serializer (FOBS), you might experience failures if your code is still using Pickle. 
To migrate the code or you experience error due to this, please refer to [Flare Object Serializer (FOBS)](https://github.com/NVIDIA/NVFlare/tree/main/nvflare/fuel/utils/fobs/README.rst)

Another type of failure is due to data types that are not supported by FOBS. By default FOBS supports some data types, if the data type (Custom Class or Class from 3rd parties)
is not part of supported FOBS data type, then you need to follow [Flare Object Serializer (FOBS)](https://github.com/NVIDIA/NVFlare/tree/main/nvflare/fuel/utils/fobs/README.rst) instructions.
Essentially, to address this type of issue, you need to do the following steps: 
* create a FobDecomposer class for the targeted data type

* Registered the newly created FobDecomposer before the data type is transmitted between client and server.  
The following examples are directly copied from [Flare Object Serializer (FOBS)](https://github.com/NVIDIA/NVFlare/tree/main/nvflare/fuel/utils/fobs/README.rst).
```
from nvflare.fuel.utils import fobs

class Simple:

    def __init__(self, num: int, name: str, timestamp: datetime):
        self.num = num
        self.name = name
        self.timestamp = timestamp


class SimpleDecomposer(fobs.Decomposer):

    @staticmethod
    def supported_type() -> Type[Any]:
        return Simple

    def decompose(self, obj) -> Any:
        return [obj.num, obj.name, obj.timestamp]

    def recompose(self, data: Any) -> Simple:
        return Simple(data[0], data[1], data[2])

```
* register the data type in FOBS before the data type is used
  you can then register the newly created FOBDecomposer
```
fobs.register(SimpleDecomposer)
```
  The decomposers must be registered in both server and client code before FOBS is used. 
  A good place for registration is the constructors for controllers and executors. It can also be done in START_RUN event handler.

* use FOBS to serialize data before you use sharable
  Custom object cannot be put in shareable directly, it must be serialized using FOBS first. Assuming custom_data contains custom type, this is how data can be stored in shareable,
```
    shareable[CUSTOM_DATA] = fobs.dumps(custom_data)
```
  On the receiving end,

```
custom_data = fobs.loads(shareable[CUSTOM_DATA])
```

This doesn't work
```
shareable[CUSTOM_DATA] = custom_data
```


### Replace TLS certificates

* Since 2.2.1, we changed the authorization model from centralized to federated authorization. This implies you can not 
use the original startup kit (which contains the old TLS certificates). You will need to cleanup the old setartup kits
re-provision your project

### Use new Project.yml template

* Since 2.2.1, we enabled federated site policies which require the new project.yml template. Please refer [default project.yml](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_system.html#default-project-yml-file)  

### new local directory
With 2.2.1, the provision will produce not only the ```startup``` directory, but a ```local``` directory. 
The resource allocation used to be in project.yml is now in resources.json and each sites/clients need to manage them separately in each location. 
you need to place/modify your own site's authorization.json and privacy.json as well if you like to change the default policies. 

The default configurations are provided in each site's local directory:
```
local
├── authorization.json.default
├── log.config.default
├── privacy.json.sample
└── resources.json.default
```
And that these defaults can be overridden by removing the default suffix and modifying the configuration as needed for the specific site.

 