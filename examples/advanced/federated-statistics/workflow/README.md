# Federated Statistics

This example illustrates the following:
* How to use the workflow communicator API to contract a federated statistics workflow: no need to write a controller.
* we can optionally turn federated statistics batch job into a streaming job
  

##  Workflow Communicator API

The workflow Communicator API only has small set methods

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
    def wait_all(self, min_responses: int, resp_max_wait_time: Optional[float]) -> Dict[str, Dict[str, FLModel]]:
        pass

    @abstractmethod
    def wait_one(self, resp_max_wait_time: Optional[float] = None) -> Tuple[str, str, FLModel]:
        pass
```


## Writing a new Workflow

With this new API writing the new workflow is really simple:

```

class FedStatistics(WF):
    def __init__(
        self,
        statistic_configs: Dict[str, dict],
        output_path: str,
        wait_time_after_1st_resp_received: float = 10,
        min_clients: Optional[int] = None,
        precision=4,
        streaming_enabled: bool = False,
        streaming_interval: float = 1,
    ):
        <skip init code>
        self.round_tasks = [StC.STATS_1st_STATISTICS, StC.STATS_2nd_STATISTICS]
        fobs_registration()

    def run(self):
        count = 1
        
        while True if self.streaming_enabled else (count <= 1):
            self.logger.info("start federated statistics run \n")
            global_statistics = {}
            client_statistics = {}
            client_features = {}
            
            for current_round, statistic_task in enumerate(self.round_tasks):

                self.logger.info(f"{current_round=}, {statistic_task} \n")

                global_statistics, client_statistics = self.statistics_task_flow(
                    current_round, global_statistics, client_statistics, client_features, statistic_task
                )

            self.logger.info("combine all clients' statistics")

            ds_stats = combine_all_statistics(
                self.statistic_configs, global_statistics, client_statistics, self.precision
            )

            save_to_json(ds_stats, self.output_path)
            self.logger.info(f"save statistics result to '{self.output_path}'\n ")

            count += 1
            if self.streaming_enabled:
                time.sleep(self.streaming_interval)
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

The federated statistics calculation involves two rounds. For each round, we call statistics_task_flow() method
with different input()

We then package the inputs into FLModel.params ( a dictionary), then call broadcast_and_wait()

```
     stats_config = FLModel(params_type=ParamsType.FULL, params=inputs)
     payload = {
            CMD: CMD_BROADCAST,
            DATA: stats_config,
            RESP_MAX_WAIT_TIME: self.wait_time_after_1st_resp_received,
            MIN_RESPONSES: self.min_clients,
            CURRENT_ROUND: current_round,
            NUM_ROUNDS: 2,
            START_ROUND: 0,
        }

    results: Dict[str, Dict[str, FLModel]] = self.flare_comm.broadcast_and_wait(payload)
```
  
## Configurations

### client-side configuration

This is the same as FLARE Client API configuration

### server-side configuration

Server side is really simple, all we need is to use WFController with newly defined workflow class

```fed_statistics_workflow.FedStatistics``` is the new 

```
{
  # version of the configuration
  format_version = 2
  task_data_filters =[]
  task_result_filters = []

  workflows = [
      {
        id = "fedStats"
        path = "nvflare.app_common.workflows.wf_controller.WFController"
        args {
            task_name = "fed_stats"
            wf_class_path = "nvflare.app_common.statistics.fed_statistics_workflow.FedStatistics",
            wf_args {
                output_path = "/tmp/fed_stats/adults.json"
                statistic_configs {
                    count = {}
                    mean = {}
                    sum = {}
                    stddev = {}
                    histogram = {
                        "*" = { bins = 20 }
                        Age = {
                            bins = 10
                            range =  [0,120]
                            }
                   }
               }
            }
        }
      }
  ]

  components = []

}



```
```nvflare.app_common.statistics.fed_statistics_workflow.FedStatistics``` is the new simple workflow class discussed above

## Run the job

assume current working directory is at ```examples/advanced/federated-statistics/workflow``` directory 
we are using the same datasets as df_stats example, you need to download the dataset first

change directory to ```examples/advanced/federated-statistics/df_stats```

run the following command to download the data

```
./prepare_data.sh
```
then go back to ```examples/advanced/federated-statistics/workflow``` directory

```
nvflare simulator -n 2 -t 2 jobs/fed_stats -w /tmp/fed_stats
```
