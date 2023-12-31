# Fed Cyclic Weight Transfer: simplified

This example illustrates  How to use the new Workflow Communication API to contract a workflow: no need to write a controller.  

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

* Workflow (Server)

```

class FedCyclic(WF):
    def __init__(
        self,
        output_path: str,
        num_rounds: int = 5,
        start_round: int = 0,
        task_name="train",
        order: str = RelayOrder.FIXED,
    ):
        super(FedCyclic, self).__init__()
        
        <skip init code>

 
    def run(self):

        self.last_model = self.init_model()

        # note: this one must be within run() method, not in the __init__() method
        # as some values are injected at runtime during run()
        self.part_sites = self.flare_comm.get_site_names()

        if len(self.part_sites) <= 1:
            raise ValueError(f"Not enough client sites. sites={self.part_sites}")

        start = self.start_round
        end = self.start_round + self.num_rounds
        for current_round in range(start, end):
            targets = self.get_relay_orders()
            relay_result = self.relay_and_wait(self.last_model, targets, current_round)

            self.logger.info(f"target sites ={targets}.")

            task_name, task_result = next(iter(relay_result.items()))
            self.last_site, self.last_model = next(iter(task_result.items()))

            self.logger.info(f"ending current round={current_round}.")
            gc.collect()

        self.save_model(self.last_model, self.output_path)
        self.logger.info("Cyclic ended.")

```

Relay_and_wait 

```

    def relay_and_wait(self, last_model: FLModel, targets: List[str], current_round):
        msg_payload = {
            MIN_RESPONSES: 1,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: self.num_rounds,
            START_ROUND: self.start_round,
            DATA: last_model,
            TARGET_SITES: targets,
        }
        # (2) relay_and_wait and wait
        results = self.flare_comm.relay_and_wait(msg_payload)
        return results
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
        id = "fed_avg"
        path = "nvflare.app_opt.pt.wf_controller.PTWFController"
        args {
            comm_msg_pull_interval = 5
            task_name = "train"
            wf_class_path = "fed_cyclic.FedCyclic",
            wf_args {
                num_rounds = 10
                output_path = "/tmp/nvflare/fedavg/mode.pth"
            }
        }
      }
  ]

  components = []

}

```


## Run the job

assume current working directory is at ```hello-cyclic-pt``` directory 

```
 nvflare simulator jobs/cyclic -w /tmp/cyclic -n 3 -t 3

```
