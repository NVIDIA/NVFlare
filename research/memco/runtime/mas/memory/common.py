# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Optional

import networkx as nx
from networkx.readwrite import json_graph


@dataclass
class StorageNameSpace:
    """
    StorageNameSpace represents a namespace for storage-related tasks,
    such as indexing and querying.

    Attributes:
        namespace (str): The identifier for this storage namespace.
        global_config (dict): A dictionary containing global configuration
                              settings for the namespace.
    """

    namespace: str
    global_config: dict

    def _index_done(self):
        pass

    def _query_done(self):
        pass


@dataclass
class AgentMessage:
    """
    AgentMessage represents a structured message exchanged between agents,
    including optional instructions and metadata.

    Attributes:
        agent_name (Optional[str]): The name of the agent sending or receiving the message.
        system_instruction (Optional[str]): Optional system-level instruction guiding the agent's behavior.
        user_instruction (Optional[str]): Optional user-level instruction or query.
        message (Optional[str]): The core message content (response or statement).
        extra_fields (dict[str, Any]): A dictionary to hold additional custom fields or metadata.
    """

    agent_name: Optional[str] = None
    system_instruction: Optional[str] = None
    user_instruction: Optional[str] = None
    message: Optional[str] = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None)


@dataclass
class StateChain:
    """
    Manages a chain of directed graph states representing the evolution of agent messages and their relationships.

    Each state is a NetworkX DiGraph, where nodes represent agent messages and edges represent connections (e.g., spatial edges)
    between agents. The class supports adding messages, moving to new states, and serializing/deserializing the chain.

    Attributes:
        chain_of_states (list[nx.DiGraph]): Internal list storing the sequence of graph states.
    """

    def __post_init__(self):
        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states = [initial_state]

    def __iter__(self) -> Iterator[nx.DiGraph]:
        return iter(self.chain_of_states)

    def __len__(self) -> int:
        return len(self.chain_of_states)

    def add_message(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:

        current_state: nx.DiGraph = self._get_current_state()

        agent_message_dict: dict = asdict(agent_message)
        node_id = self._generate_node_id(agent_message.agent_name)
        current_state.add_node(node_id, **agent_message_dict)

        for up_node_id in upstream_agent_ids:
            if not current_state.has_node(up_node_id):
                raise ValueError("Upstream node does not exist.")
            current_state.add_edge(up_node_id, node_id, edge_type="spatial")
        return node_id

    def move_state(self, action: str, observation: str, **args) -> None:

        current_state: nx.DiGraph = self._get_current_state()
        current_state.graph.update({"action": action, "observation": observation, **args})

        initial_state = nx.DiGraph()
        initial_state.graph["name_counter"] = {}
        self._chain_of_states.append(initial_state)

    def get_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError("Index out of range.")
        return self.chain_of_states[idx]

    def pop_state(self, idx: int) -> nx.Graph:
        if idx >= len(self.chain_of_states) or idx < -len(self.chain_of_states):
            raise ValueError("Index out of range.")
        return self.chain_of_states.pop(idx)

    @property
    def chain_of_states(self) -> list[nx.Graph]:
        return self._chain_of_states[:-1]

    def _generate_node_id(self, agent_name: str) -> str:
        current_state: nx.DiGraph = self._get_current_state()
        name_counter = current_state.graph["name_counter"]

        if agent_name not in name_counter:
            name_counter[agent_name] = 0
        else:
            name_counter[agent_name] += 1
        return f"{agent_name}-{name_counter[agent_name]}"

    def _get_current_state(self) -> nx.DiGraph:
        return self._chain_of_states[-1]

    @staticmethod
    def to_str(state_chain: "StateChain") -> str:
        return json.dumps([json_graph.node_link_data(state) for state in state_chain])

    @staticmethod
    def from_str(state_chain_str: str) -> "StateChain":
        state_chain = StateChain()
        if not state_chain_str:
            return state_chain
        try:
            payload = json.loads(state_chain_str)
            state_chain._chain_of_states = [json_graph.node_link_graph(state_data) for state_data in payload]
        except Exception:
            # Fallback for legacy/invalid formats.
            state_chain = StateChain()
        return state_chain


@dataclass
class MASMessage:
    """
    Represents a multi-agent system (MAS) message that encapsulates the main task,
    its description, the trajectory of task actions and observations, and the
    associated chain of states tracking the message evolution.

    Attributes:
        task_main (str): The main task or objective description.
        task_description (Optional[str]): Additional details or description of the task.
        task_trajectory (Optional[str]): A textual log of the sequence of actions and observations. Defaults to a prompt format.
        label (Optional[bool]): An optional label or flag associated with the task.
        chain_of_states (StateChain): A StateChain instance tracking the evolution of states/messages in the task.
        extra_fields (dict[str, Any]): A dictionary for storing additional arbitrary fields related to the message.
    """

    task_main: str
    task_description: Optional[str] = None
    task_trajectory: Optional[str] = "\n\n>"
    label: Optional[bool] = None
    chain_of_states: StateChain = field(default_factory=StateChain, repr=False)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)

    def add_message_to_current_state(self, agent_message: AgentMessage, upstream_agent_ids: list[str]) -> str:
        return self.chain_of_states.add_message(agent_message, upstream_agent_ids)

    def move_state(self, action: str, observation: str, **args) -> None:
        self.task_trajectory += f"{action}\n{observation}\n>"
        self.chain_of_states.move_state(action, observation, **args)

    def add_extra_field(self, key: str, value: Any):
        self.extra_fields[key] = value

    def get_extra_field(self, key: str) -> Optional[Any]:
        return self.extra_fields.get(key, None)

    @staticmethod
    def to_dict(mas_message: "MASMessage") -> dict[str, str]:
        return {
            "task_main": mas_message.task_main,
            "task_description": mas_message.task_description,
            "task_trajectory": mas_message.task_trajectory,
            "label": mas_message.label,
            "extra_fields": json.dumps(mas_message.extra_fields),
            "state_chain": StateChain.to_str(mas_message.chain_of_states),
        }

    @staticmethod
    def from_dict(message_dict: dict) -> "MASMessage":
        return MASMessage(
            task_main=message_dict.get("task_main"),
            task_description=message_dict.get("task_description"),
            task_trajectory=message_dict.get("task_trajectory"),
            label=message_dict.get("label"),
            extra_fields=json.loads(message_dict.get("extra_fields", "{}")),
            chain_of_states=StateChain.from_str(message_dict.get("state_chain")),
        )
