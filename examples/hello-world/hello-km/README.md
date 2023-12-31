# Kaplan-Meier Analysis

This example illustrates two features:
* How to perform Kaplan-Meirer Survival Analysis in federated setting
* How to use the new Flare Communicator API to contract a workflow: no need to write a controller.  

## FLARE Workflow Communicator API

The Flare workflow Communicator API only has small set methods

```
class WFCommAPISpec(ABC):
    @abstractmethod
    def broadcast_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def broadcast(self, msg_payload):
        pass

    @abstractmethod
    def send(self, msg_payload: Dict):
        pass

    @abstractmethod
    def send_and_wait(self, msg_payload: Dict):
        pass

    @abstractmethod
    def get_site_names(self):
        pass

    @abstractmethod
    def wait(self, min_responses):
        pass
```


## Writing a new Workflow

With this new API writing the new workflow is really simple: 

For example for Kaplan-Meier Analysis, we could write a new workflow like this: 

```

class KM(WF):
    def __init__(self, min_clients: int, output_path: str):
        super(KM, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_path = output_path
        self.min_clients = min_clients
        self.num_rounds = 1

    def run(self):
        results = self.start_km_analysis()
        global_res = self.aggr_km_result(results)
        self.save(global_res, self.output_path)

```

The base class ```WF``` is define as

```

class WF(ABC):

    def __init__(self):
        self.flare_comm: Optional[WFCommAPI] = None

    def setup_wf_comm_api(self, flare_comm: WFCommAPI):
        self.flare_comm = flare_comm

    @abstractmethod
    def run(self):
        raise NotImplementedError

```
has two expectations:
* Make sure user define ```run()``` method
* make sure a class field of WFCommAPI and be able to dynamically populated at runtime
  via setup_wf_comm_api() method


for Kaplan-Meier analysis, it literal involves

* start the analysis --> ask all clients to perform local KM analysis, then wait for results 
* then aggregate the result to obtain gloabl results
* save the result

We only need to one_round trip from server --> client, client --> server

Let's define the start_km_analysis()

```

    def start_km_analysis(self):
        self.logger.info("send kaplan-meier analysis command to all sites \n")

        msg_payload = {
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: 1,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: 1,
            DATA: {},
        }

        results = self.flare_comm.broadcast_and_wait(msg_payload)
        return results

```

looks like to simply call send broadcast command, then just get the results.

## Configurations

### client-side configuration

This is the same as FLARE Client API configuration

### server-side configuration

  Server side controller is really simple, all we need is to use WFController with newly defined workflow class
 

```
{
  # version of the configuration
  format_version = 2
  task_data_filters =[]
  task_result_filters = []

  workflows = [
      {
        id = "km"
        path = "nvflare.app_common.workflows.wf_controller.WFController"
        args {
            task_name = "train"
            wf_class_path = "kaplan_meier.KM",
            wf_args {
                min_clients = 2
                output_path = "/tmp/nvflare/km/km.json"
            }
        }
      }
  ]

  components = []

}


```


## Run the job

assume current working directory is at ```hello-km``` directory 

```
nvflare simulator -n 2 -t 2 jobs/kaplan-meier -w /tmp/km
```


## Display Result

Once finished the results will be written to the output_path defined about. 
We can copy the result to the demo directory and start notebook

```
cp /tmp/nvflare/km/km.json demo/.

jupyter lab demo/km.ipynb 

```
![KM survival curl](km_survival_curve.png)
