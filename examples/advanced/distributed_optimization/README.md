# P2P Distributed Optimization algorithms with NVFlare

In this example we show how to exploit the lower-level NVFlare APIs to implement and run P2P distributed optimization algorithms. The aim here is twofold: on one hand we provide a few [examples](#examples) showing how to directly use the `nvflare.app_opt.p2p` API to run distributed optimization algorithms, on the other hand we provide a [walkthrough](#implementation-walkthrough) of the actual implementation of the APIs in `nvflare.app_opt.p2p` to show how to exploit lower-level NVFlare APIs for advanced use-cases.


## Examples
The following algorithms are currently implemented in `nvflare.app_opt.p2p`:
- Consensus algorithm - initially published in [DeGroot, M. H. (1974). Reaching a Consensus. Journal of the American Statistical Association, 69(345), 118–121.](https://pages.ucsd.edu/~aronatas/project/academic/degroot%20consensus.pdf)
- Distributed (stochastic) gradient descent [Tsitsiklis, J., Bertsekas, D., & Athans, M. (1986). Distributed asynchronous deterministic and stochastic gradient optimization algorithms. IEEE transactions on automatic control, 31(9), 803-812.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1104412) and [Sundhar Ram, S., Nedić, A., & Veeravalli, V. V. (2010). Distributed stochastic subgradient projection algorithms for convex optimization. Journal of optimization theory and applications, 147, 516-545.](https://arxiv.org/pdf/0811.2595)
- (Stochastic) gradient tracking [Pu, S., & Nedić, A. (2021). Distributed stochastic gradient tracking methods. Mathematical Programming, 187(1), 409-457.](https://arxiv.org/pdf/1805.11454)
- GTAdam [Carnevale, G., Farina, F., Notarnicola, I., & Notarstefano, G. (2022). GTAdam: Gradient tracking with adaptive momentum for distributed online optimization. IEEE Transactions on Control of Network Systems, 10(3), 1436-1448.](https://ieeexplore.ieee.org/abstract/document/9999485)


In this repo we provide the following examples:
- [1-consensus](./1-consensus/) - a simple consensus algorithm to compute the average of a set of numbers
- [2-two_moons](./2-two_moons/) - different distributed optimization algorithms solving the two moons classification problem
- [3-mnist](./3-mnist/) - different distributed optimization algorithms training local models to classify MNIST images in a heavily unbalanced setting


## Implementation walkthrough
Let's now walk through how to use NVFlare to implement custom peer-to-peer (P2P) algorithms, opening the road to easily implement custom distributed optimization and swarm learning workflows.
Specifically, we'll delve into using some lower-level NVFlare APIs to create a controllers and executors, which serve as the backbone for orchestrating communication and computation across different nodes (clients) in a distributed setup. 
As an example, we'll demonstrate how to implement a consensus algorithm using these components.

As said, the final implementation is in the `nvflare.app_opt.p2p` module - we'll refer to the specific files along the notebook.

### Introduction

In peer-to-peer (P2P) algorithms, being distributed optimization or decentralized federated learning algorithms, clients communicate directly with each other without relying on a central server to aggregate updates. Implementing P2P algorithms usually requires careful orchestration to handle communication, synchronization, and data exchange among clients.

Thankfully, NVFlare natively provides the primitives to easily build such a system.

#### Exploiting NVFlare low-level APIs
NVFlare natively supports various communication and orchestration patterns, including peer-to-peer interactions, which are crucial for decentralized algorithms.

To implement custom P2P/distributed optimization algorithms, we'll delve into its lower level APIs to build a framework facilitate building P2P algorithms. In particular, we'll use

- [Controllers](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.impl.controller.html#module-nvflare.apis.impl.controller): Server-side components that manage job execution and orchestrate tasks.
- [Executors](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.apis.executor.html#module-nvflare.apis.executor): Client-side components that perform computations and handle tasks received from the controller.
- [Messages via aux channes](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.private.aux_runner.html#nvflare.private.aux_runner.AuxRunner.send_aux_request): Custom messages that enable direct communication between clients.

#### What to expect
We'll start by defining a way to easily define and share configurations across the network. Then we'll implement a base controller and executor, serving as the backbone to implement arbitrary p2p algorithms. Finally we'll build upon the base executor to implement a specific algorithm, the Consensus algorithm.

### Pythonic configs

Before we dive into the controller and executor, let's start by creating a Pythonic way to define the configuration of our network. We'll start by simply defining a `Node` (i.e. a client in the network), its `Neighbors` (i.e. other clients with which a client communicates, each with a weight) and combine them to define a `Network` (i.e. a network of clients with neighbors).

```python
from dataclasses import dataclass, field

@dataclass
class Neighbor:
    id: int | str
    weight: float | None = None

@dataclass
class Node:
    id: int | str
    neighbors: list[Neighbor] = field(default_factory=list)

@dataclass
class Network:
    nodes: list[Node] = field(default_factory=list)
```

Then we'll define a global and a local config objects to be passed to the controller and executors respectively.

```python
@dataclass
class Config:
    network: Network
    extra: dict = field(default_factory=dict)

@dataclass
class LocalConfig:
    neighbors: list[Neighbor]
    extra: dict = field(default_factory=dict)
```

The `extra` parameter can be used to pass additional parameters, usually specific for the various algorithms. 

To actual implementation of the objects above can be found in `nvflare/app_opt/p2p/types/__init__.py` (you'll see they'll have the `__dict__` and `__post_init__` methods defined facilitate serializing and deserializing them, which is needed for NVFlare).

Here's an example of a ring network with 3 clients, running an algorithm for 100 iterations:
```shell
Config(
    extra={"iterations":100},
    network=Network(
        nodes=[
            Node(
                id='site-1',
                neighbors=[
                    Neighbor(id='site-2', weight=0.1),
                ]
            ),
            Node(
                id='site-2',
                neighbors=[
                    Neighbor(id='site-3', weight=0.1),
                ]
            ),
            Node(
                id='site-3',
                neighbors=[
                    Neighbor(id='site-1', weight=0.1),
                ]
            ),
        ]
    )
)
```

### The controller

In NVFlare, a `Controller` is a server-side component that manages the job execution and orchestration of tasks. Here, since we're running a P2P algorithm, we'll implement a custom controller whose main job is to load and broadcast the network configuration, and initiate/terminate the execution of a P2P distributed optimization algorithm. Let's call it `DistOptController`. As a subclass of `Controller`, it must implement 3 methods:

- `start_controller` which is called at the beginning of the run
- `control_flow` defining the main control flow of the controller (in this case, broadcasting the configuration and asking clients to run the algorithm)
- `stop_controller`, called at the end of the run

```python
from nvflare.apis.impl.controller import Controller

class DistOptController(Controller):

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Broadcast configuration to clients
        ...

        # Run the algorithm
        ...

    def start_controller(self, fl_ctx: FLContext):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        pass
```

We won't do anything fancy during the start and stop phase, so let's focus on the `control_flow` and implement the two steps. To do so, we first need to override the `__init__` method to take a `Config` object as an argument. 

```python
from nvflare.app_opt.p2p.types import Config

class DistOptController(Controller):
    def __init__(
        self,
        config: Config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config

    ... 
```

Now, in the `control_flow` method we can send the local configurations to each client and, once they receive them, ask them to run the algorithm. We'll do so by sending `Task`s to each client. In NVFlare, a `Task` is a piece of work that is assigned by the Controller to client workers. Depending on how the task is assigned (broadcast, send, or relay), the task will be performed by one or more clients.

In fact, on one hand, we'll use the `send_and_wait` method to send the `"config"` task to each client, since each client will potentially have a different config (because of different neighbors); on the other hand, to run the algorith, we'll use the `broadcast_and_wait`, which broadcasts the same `"run_algorithm"` task to all clients and waits for all clients to respond/complete the task. As you see, each task specifies a `name` - in this case, `"config"` and `"run_algorithm"` - let's remember those as they'll need to be the same in the control flow of each client.


```python
from nvflare.apis.controller_spec import Task
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal

class DistOptController(Controller):

    ...

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # Send network config (aka neighors info) to each client
        for node in self.config.network.nodes:
            task = Task(
                name="config",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={"neighbors": [n.__dict__ for n in node.neighbors]},
                ).to_shareable(),
            )
            self.send_and_wait(task=task, targets=[node.id], fl_ctx=fl_ctx)

        # Run algorithm (with extra params if any passed as data)
        targets = [node.id for node in self.config.network.nodes]
        self.broadcast_and_wait(
            task=Task(
                name="run_algorithm",
                data=DXO(
                    data_kind=DataKind.APP_DEFINED,
                    data={key: value for key, value in self.config.extra.items()},
                ).to_shareable(),
            ),
            targets=targets,
            min_responses=0,
            fl_ctx=fl_ctx,
        )
    
    ... 
```

And that's it, our `DistOptController` is ready. The complete implementation of the `DistOptController` can be found in `nvflare/app_opt/p2p/controllers/dist_opt_controller.py`.

### The executor

Now that we have our `DistOptController`, it's time to take care of the actual execution of the algorithm at the client level - we'll build on top of the NVFlare `Executor` to do to that.

In NVFlare, an `Executor` is a client-side component that handles tasks received from the controller and executes them. For our purposes we'll need our executor to be able to do a few things:
- receive the config from the server/controller
- communicate with its neighbors and send/receive messages to/from them
- run the algorithm

For the moment, we'll focus on synchronous algorithms only, meaning that the clients will need to run the iterations of an algorithm in a synchronous way. Let' call our executor `SyncAlgorithmExecutor`.
The only method that must be implemented in this case is the `execute` method, which takes the `task_name` and `shareable` sent from the controller as inputs.

```python
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply


class SyncAlgorithmExecutor(Executor):

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "config":
            # TODO: receive and store config
            ...
            return make_reply(ReturnCode.OK)

        elif task_name == "run_algorithm":
            # TODO: run the algorithm
            return make_reply(ReturnCode.OK)
        else:
            self.log_warning(fl_ctx, f"Unknown task name: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
```

Let's focus on the execution of the `"config"` task - for this we can just create some attributes to store the config and the neighbors (and the local weight computed from them).

```python
from nvflare.apis.dxo import from_shareable
from nvdo.types import LocalConfig, Neighbor


class SyncAlgorithmExecutor(Executor):
    def __init__(self):
        super().__init__()

        self.config = None
        self._weight = None
        self.neighbors: list[Neighbor] = []


    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "config":
            # Receive and store config
            self.config = LocalConfig(**from_shareable(shareable).data)
            self.neighbors = self.config.neighbors
            self._weight = 1.0 - sum([n.weight for n in self.neighbors])
            return make_reply(ReturnCode.OK)

        elif task_name == "run_algorithm":
            # TODO: run the algorithm
            return make_reply(ReturnCode.OK)
        else:
            self.log_warning(fl_ctx, f"Unknown task name: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)
```

That was relatively easy - so, now to the slightly more challenging part of letting clients communicate with each other.
To do that we'll do a few things:
- we'll use the `send_aux_request` method to let a client send a message to its neighbors
- we'll need to register a callback to handle received messages (via the `register_aux_message_handler` function) and add an attribute `neighbors_values` to store received values. We'll call the callback `_handle_neighbor_value` and the registration will be done in the `handle_event` method at start time (i.e., when receiving the `EventType.START_RUN` event). Other events can be handled in the same way if needed.
- we'll use threading events and locks to synchronize the execution of the algorithm (making each client, when sending a message, wait to have received the messages of all its neighbors before sending the next message)
- we'll add two methods, `_from_message` and `_to_message` to convert between the message exchange formats (which will need to be overridden in subclasses, based on the algorithm)

The main message exchange will be done in the `_exchange_values` function.

```python
import threading
from abc import abstractmethod
from collections import defaultdict

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReservedKey
from nvflare.apis.signal import Signal


class SyncAlgorithmExecutor(Executor):
    def __init__(self):
        super().__init__()
        ... # other attributes

        self.neighbors_values = defaultdict(dict)

        self.sync_waiter = threading.Event()
        self.lock = threading.Lock()


    def _exchange_values(self, fl_ctx: FLContext, value: any, iteration: int):
        engine = fl_ctx.get_engine()

        # Clear the event before starting the exchange
        self.sync_waiter.clear()

        # Send message to neighbors
        _ = engine.send_aux_request(
            targets=[neighbor.id for neighbor in self.neighbors],
            topic="send_value",
            request=DXO(
                data_kind=DataKind.METRICS,
                data={
                    "value": self._to_message(value),
                    "iteration": iteration,
                },
            ).to_shareable(),
            timeout=10,
            fl_ctx=fl_ctx,
        )

        # check if all neighbors sent their values
        if len(self.neighbors_values[iteration]) < len(self.neighbors):
            # if not, wait for them, max 10 seconds
            if not self.sync_waiter.wait(timeout=10):
                self.system_panic("failed to receive values from all neighbors", fl_ctx)
                return

    def _handle_neighbor_value(
        self, topic: str, request: Shareable, fl_ctx: FLContext
    ) -> Shareable:
        sender = request.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default=None)
        data = from_shareable(request).data
        iteration = data["iteration"]

        with self.lock:
            self.neighbors_values[iteration][sender] = self._from_message(data["value"])
            # Check if all neighbor values have been received
            if len(self.neighbors_values[iteration]) >= len(self.neighbors):
                self.sync_waiter.set()  # Signal that we have all neighbor values
        return make_reply(ReturnCode.OK)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()

            engine.register_aux_message_handler(
                topic="send_value", message_handle_func=self._handle_neighbor_value
            )

    def _to_message(self, x):
        return x

    def _from_message(self, x):
        return x

```

That's it for the synchronous message exchange. Notice that `neighbors_values` needs to maintain a dictionary of received values per iteration. 
This is because, different parts of a network may be at different iterations of the algorithm (plus or minus 1 at most) - this means that I could receive a message from a neighbor valid for iteration `t+1` when I'm still at iteration `t`. Since that message won't be sent again, I need to store it. To avoid the `neighbors_values` to grow indefinitely, we'll delete its content at iteration `t` after having consumed its values and moving to the next iteration in the algorithm. We'll see that in the next section.

Moving forward, now that we have a way to store the config and exchange messages with the neighbors, we can move on to implementing the algorithmic part. For this base `SyncAlgorithmExecutor`, we'll just implement the main logic in the `execute` method, based on an abstract `run_algorithm` to be overridden by each specific algorithm.

```python
class SyncAlgorithmExecutor(Executor):
    
    ...
    
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ):
        if task_name == "config":
            # Receive topology from the server
            self._load_config(shareable, fl_ctx)
            return make_reply(ReturnCode.OK)

        elif task_name == "run_algorithm":
            self.run_algorithm(fl_ctx, shareable, abort_signal)
            return make_reply(ReturnCode.OK)
        else:
            self.log_warning(fl_ctx, f"Unknown task name: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    @abstractmethod
    def run_algorithm(
        self, fl_ctx: FLContext, shareable: Shareable, abort_signal: Signal
    ):
        """Executes the algorithm"""
        pass
    
    ...
```

And that's all. The full implementation is in `nvflare/app_opt/p2p/executors/sync_executor.py` - note that the implementation of the `SyncAlgorithmExecutor` in `nvflare.app_opt.p2p` is a subclass of `BaseDistOptExecutor`, defined in `nvflare/app_opt/p2p/executors/base_dist_opt_executor.py`. It contains a few additional attributes (namely `self.id` and `self.client_name`) to identify the client, which are potentially useful in algorithms, and two additional methods `_pre_algorithm_run` and `_post_algorithm_run` to be overridden by each specific algorithm to execute some code before and after the algorithm execution, respectively.

### An example: the `ConsensusExecutor`

Now that we have built all the main foundations, we can easily implement any custom P2P algorithm. For example, let's implement a slightly simplified version of the `ConsensusExecutor` that will be used in the next section and whose full implementation is in `nvflare/app_opt/p2p/executors/consensus.py`.

```python
import torch

class ConsensusExecutor(SyncAlgorithmExecutor):

    def __init__(
        self,
        initial_value: float | None = None,
    ):
        super().__init__()
        if initial_value is None:
            initial_value = random.random()
        self.current_value = initial_value
        self.value_history = [self.current_value]

    def run_algorithm(self, fl_ctx, shareable, abort_signal):
        iterations = from_shareable(shareable).data["iterations"]

        for iteration in range(iterations):
            # 1. exchange values
            self._exchange_values(
                fl_ctx, value=self.current_value, iteration=iteration
            )

            # 2. compute new value
            current_value = self.current_value * self._weight
            for neighbor in self.neighbors:
                current_value += (
                    self.neighbors_values[iteration][neighbor.id] * neighbor.weight
                )

            # 3. store current value
            self.current_value = current_value

            # free memory that's no longer needed
            del self.neighbors_values[iteration]

```

As you can see, it's basically just a matter of implementing the algorithm logic in the `run_algorithm` method. Feel free to explore the `nvflare.app_opt.p2p` module to see how other algorithms are implemented.