from dataclasses import dataclass, field


@dataclass
class Neighbor:
    """Represents a neighbor in the network."""

    id: int | str
    weight: float | None = None

    @property
    def __dict__(self):
        return {"id": self.id, "weight": self.weight}


@dataclass
class Node:
    """Represents a node in the network."""

    id: int | str | None = None
    neighbors: list[Neighbor] = field(default_factory=list)

    @property
    def __dict__(self):
        return {
            "id": self.id,
            "neighbors": [neighbor.__dict__ for neighbor in self.neighbors],
        }

    def __post_init__(self):
        new_neighbors = []
        for neighbor in self.neighbors:
            if isinstance(neighbor, dict):
                new_neighbors.append(Neighbor(**neighbor))
            else:
                new_neighbors.append(neighbor)
        self.neighbors = new_neighbors


@dataclass
class Network:
    """Represents a network."""

    nodes: list[Node] = field(default_factory=list)

    @property
    def __dict__(self):
        return {"nodes": [node.__dict__ for node in self.nodes]}

    def __post_init__(self):
        new_nodes = []
        for node in self.nodes:
            if isinstance(node, dict):
                new_nodes.append(Node(**node))
            else:
                new_nodes.append(node)
        self.nodes = new_nodes


@dataclass
class Config:
    network: Network

    extra: dict = field(default_factory=dict)

    @property
    def __dict__(self):
        return {"extra": self.extra, "network": self.network.__dict__}

    def __post_init__(self):
        if isinstance(self.network, dict):
            self.network = Network(**self.network)


@dataclass
class LocalConfig:
    neighbors: list[Neighbor]

    extra: dict = field(default_factory=dict)

    @property
    def __dict__(self):
        return {"neighbors": self.neighbors__dict__, "extra": self.extra}

    def __post_init__(self):
        new_neighbors = []
        for neighbor in self.neighbors:
            if isinstance(neighbor, dict):
                new_neighbors.append(Neighbor(**neighbor))
            else:
                new_neighbors.append(neighbor)
        self.neighbors = new_neighbors
