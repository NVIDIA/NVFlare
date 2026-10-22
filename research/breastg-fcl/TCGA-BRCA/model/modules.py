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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("GFedCL")


# -----------------------------
# For GFedCL - Updated for TCGA-BRCA
# -----------------------------
class GNet(nn.Module):
    """Generate replay latents from noise, labels, and a client graph row."""

    def __init__(self, opt):
        super().__init__()
        self.num_clients = int(opt.num_clients)
        self.graph_dim = self.num_clients
        self.hidden_dim = int(opt.nh)
        self.output_dim = self.hidden_dim
        self.num_classes = int(opt.num_classes)
        self.noise_dim = int(getattr(opt, "noise_dim", opt.nh))
        if min(self.graph_dim, self.hidden_dim, self.num_classes, self.noise_dim) < 1:
            raise ValueError("Generator dimensions and class count must be positive")

        condition_dim = max(1, self.hidden_dim // 4)
        self.label_embedding = nn.Embedding(self.num_classes, condition_dim)
        self.graph_processor = nn.Sequential(
            nn.Linear(self.graph_dim, condition_dim),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(self.noise_dim + 2 * condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
        )

    def _graph_rows(self, graph_row, batch_size, reference):
        if graph_row is None:
            raise ValueError("Generator requires a client graph row")
        graph = torch.as_tensor(graph_row, device=reference.device, dtype=reference.dtype)
        if graph.ndim == 1:
            graph = graph.unsqueeze(0)
        if graph.ndim != 2 or graph.shape[1] != self.graph_dim or graph.shape[0] not in (1, batch_size):
            raise ValueError(
                f"Graph rows must have shape [{self.graph_dim}], [1, {self.graph_dim}], "
                f"or [{batch_size}, {self.graph_dim}]; got {tuple(graph.shape)}"
            )
        if not torch.isfinite(graph).all():
            raise ValueError("Graph rows contain non-finite values")
        return graph.expand(batch_size, -1)

    def _label_indices(self, labels, batch_size, device):
        if labels is None:
            raise ValueError("Generator requires class labels")
        labels = torch.as_tensor(labels, device=device)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if labels.is_complex() or not torch.isfinite(labels).all():
            raise ValueError("Class labels must be finite real values")
        if labels.ndim == 2 and tuple(labels.shape) == (batch_size, self.num_classes):
            labels = labels.argmax(dim=1)
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError(f"Labels must have shape [{batch_size}] or " f"[{batch_size}, {self.num_classes}]")
        if labels.is_floating_point() and not torch.equal(labels, labels.round()):
            raise ValueError("Class indices must be integers")
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError(f"Class indices must be in [0, {self.num_classes})")
        return labels.long()

    def forward(self, noise, labels, graph_row):
        """Return ``[batch, nh]`` latents; labels are indices or class-score rows."""
        reference = self.net[0].weight
        noise = torch.as_tensor(noise, device=reference.device, dtype=reference.dtype)
        if noise.ndim == 1:
            noise = noise.unsqueeze(0)
        if noise.ndim != 2 or noise.shape[0] < 1 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"Noise must have shape [batch, {self.noise_dim}]")
        if not torch.isfinite(noise).all():
            raise ValueError("Noise contains non-finite values")
        batch_size = noise.shape[0]
        graph = self._graph_rows(graph_row, batch_size, reference)
        indices = self._label_indices(labels, batch_size, reference.device)
        conditions = torch.cat(
            [noise, self.label_embedding(indices), self.graph_processor(graph)],
            dim=1,
        )
        return self.net(conditions)


class FeatureEncoder(nn.Module):
    """Encode RNA-seq features and the client graph row without target labels."""

    def __init__(self, opt):
        super().__init__()
        self.input_dim = int(opt.input_dim)
        self.hidden_dim = int(opt.nh)
        self.graph_dim = int(opt.num_clients)
        if min(self.input_dim, self.hidden_dim, self.graph_dim) < 1:
            raise ValueError("Encoder input, latent, and graph dimensions must be positive")
        graph_hidden_dim = max(1, self.hidden_dim // 4)

        self.data_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2) if not opt.no_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(opt.p),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if not opt.no_bn else nn.Identity(),
            nn.ReLU(),
        )

        self.graph_processor = nn.Sequential(
            nn.Linear(self.graph_dim, graph_hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + graph_hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, graph_row):
        """Return ``[batch, nh]`` features conditioned on a required graph row."""
        reference = self.data_encoder[0].weight
        data = torch.as_tensor(x, device=reference.device, dtype=reference.dtype)
        if data.ndim == 1:
            data = data.unsqueeze(0)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != self.input_dim:
            raise ValueError(f"RNA-seq input must have shape [batch, {self.input_dim}]")
        if not torch.isfinite(data).all():
            raise ValueError("RNA-seq input contains non-finite values")
        if graph_row is None:
            raise ValueError("Encoder requires a client graph row")
        graph = torch.as_tensor(graph_row, device=reference.device, dtype=reference.dtype)
        if graph.ndim == 1:
            graph = graph.unsqueeze(0)
        batch_size = data.shape[0]
        if graph.ndim != 2 or graph.shape[1] != self.graph_dim or graph.shape[0] not in (1, batch_size):
            raise ValueError(
                f"Graph rows must have shape [{self.graph_dim}], [1, {self.graph_dim}], "
                f"or [{batch_size}, {self.graph_dim}]; got {tuple(graph.shape)}"
            )
        if not torch.isfinite(graph).all():
            raise ValueError("Graph rows contain non-finite values")
        graph = graph.expand(batch_size, -1)
        combined = torch.cat(
            [self.data_encoder(data), self.graph_processor(graph)],
            dim=1,
        )
        return self.fusion(combined)


class GraphDNet(nn.Module):
    """
    Graph Discriminator - reconstructs graph embedding from encoder latent space
    """

    def __init__(self, opt):
        super(GraphDNet, self).__init__()
        self.input_dim = opt.nh
        self.hidden_dim = opt.nh
        self.output_dim = opt.nt

        # Simple network with minimal operations
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # Clone input to avoid in-place modifications
        x_copy = x.clone()

        # Always use 2D tensors for network operations
        if x_copy.dim() > 2:
            batch_shape = x_copy.shape[:-1]
            x_flat = x_copy.reshape(-1, x_copy.size(-1))
            output = self.net(x_flat)
            # Reshape back to original batch dimensions
            return output.reshape(*batch_shape, self.output_dim)
        else:
            return self.net(x_copy)


class PredNet(nn.Module):
    """
    Prediction Network - classifies encoded features
    Updated for TCGA-BRCA
    """

    def __init__(self, opt):
        super(PredNet, self).__init__()
        self.input_dim = opt.nh
        self.hidden_dim = opt.nh
        self.num_classes = opt.num_classes

        # Enhanced classifier for more complex dataset
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.num_classes),
        )

    def forward(self, x, return_softmax=False):
        # Clone input to avoid in-place operations
        x_copy = x.clone()

        # Always use 2D tensors for network operations
        original_shape = x_copy.shape
        if x_copy.dim() > 2:
            x_flat = x_copy.reshape(-1, x_copy.size(-1))
        else:
            x_flat = x_copy

        # Forward pass
        logits = self.net(x_flat)

        # Get softmax probabilities
        softmax_probs = F.softmax(logits, dim=1)

        # Preserve finite log probabilities and gradients for extreme logits.
        log_probs = F.log_softmax(logits, dim=1)

        # Reshape outputs if needed
        if x_copy.dim() > 2:
            new_shape = original_shape[:-1] + (self.num_classes,)
            log_probs = log_probs.reshape(*new_shape)
            softmax_probs = softmax_probs.reshape(*new_shape)

        if return_softmax:
            return log_probs, softmax_probs
        else:
            return log_probs


# -----------------------------
class AdditiveAttentionHead(nn.Module):
    """GFedCL's additive GAT score, without dropout on graph probabilities."""

    def __init__(self, input_dim, head_dim, temperature=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, head_dim))
        self.a = nn.Parameter(torch.empty(2 * head_dim, 1))
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        self.temperature = temperature

    def forward(self, features):
        projected = features @ self.W
        head_dim = projected.shape[1]
        # a^T [Wh_i || Wh_j], without materializing N x N x 2d pairs.
        source = projected @ self.a[:head_dim]
        target = projected @ self.a[head_dim:]
        scores = F.leaky_relu(source + target.T, negative_slope=0.2)
        return torch.softmax(scores / self.temperature, dim=1)


class SpatialAttention(nn.Module):
    """Morphology encoder and multi-head additive attention adapted from GFedCL."""

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64, n_heads=4, dropout=0.2, temperature=1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # LayerNorm also supports a single client; GFedCL used BatchNorm.
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.heads = nn.ModuleList(
            [AdditiveAttentionHead(embedding_dim, embedding_dim // 2, temperature) for _ in range(n_heads)]
        )

    def forward(self, summaries):
        features = self.encoder(summaries)
        return torch.stack([head(features) for head in self.heads]).mean(dim=0)


class TemporalAttention(nn.Module):
    """Scaled query/key attention over concatenated DCE task summaries."""

    def __init__(self, input_dim, key_dim=64, temperature=1.0):
        super().__init__()
        self.query = nn.Linear(input_dim, key_dim, bias=False)
        self.key = nn.Linear(input_dim, key_dim, bias=False)
        self.scale = key_dim**0.5 * temperature

    def forward(self, summaries):
        scores = self.query(summaries) @ self.key(summaries).T / self.scale
        return torch.softmax(scores, dim=1)


class BreastGraphGenerator(nn.Module):
    """Neural scores for the BreastG-FCL Section III-C graph equations.

    Spatial attention adapts GFedCL's 128/64 encoder and four additive GAT
    heads. Temporal attention uses query/key projections of DCE summaries,
    rather than GFedCL's fixed cosine score over historical graph rows.

    ``forward`` is differentiable. ``learn`` is a compatibility inference
    entry point: like the local GFedCL reference, it does not optimize the
    attention parameters. No attention training objective is specified in
    BreastG-FCL, so we do not introduce one implicitly.
    """

    def __init__(self, opt, spatial_dim=None, temporal_dim=None):
        super().__init__()
        self.num_clients = int(opt.num_clients)
        self.temporal_window = int(opt.temporal_window)
        self.temperature = float(opt.attention_temperature)
        self.epsilon = float(opt.graph_epsilon)
        self.hidden_dim = int(getattr(opt, "gat_hidden_dim", 128))
        self.embedding_dim = int(getattr(opt, "gat_embedding_dim", 64))
        self.n_heads = int(getattr(opt, "gat_heads", 4))
        self.dropout = float(getattr(opt, "gat_dropout", 0.2))
        self.spatial_dim = self._feature_dim(opt, spatial_dim, "spatial")
        self.temporal_dim = self._feature_dim(opt, temporal_dim, "temporal")
        if (
            min(
                self.num_clients,
                self.temporal_window,
                self.hidden_dim,
                self.n_heads,
                self.spatial_dim,
                self.temporal_dim,
            )
            < 1
        ):
            raise ValueError("Client count, window, and attention dimensions must be positive")
        if self.embedding_dim < 2:
            raise ValueError("gat_embedding_dim must be at least 2")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("attention_temperature must be finite and positive")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("graph_epsilon must be finite and non-negative")
        if not 0 <= self.dropout < 1:
            raise ValueError("gat_dropout must be in [0, 1)")

        self.spatial_attention = SpatialAttention(
            self.spatial_dim,
            self.hidden_dim,
            self.embedding_dim,
            self.n_heads,
            self.dropout,
            self.temperature,
        )
        self.temporal_attention = TemporalAttention(
            self.temporal_window * self.temporal_dim,
            self.embedding_dim,
            self.temperature,
        )
        self.temporal_history = []
        self.last_spatial_attention = None
        self.last_temporal_patterns = None
        self.last_temporal_window = None

    @staticmethod
    def _feature_dim(opt, explicit_dim, name):
        if explicit_dim is None:
            explicit_dim = getattr(opt, f"{name}_dim", None)
        if explicit_dim is not None:
            return int(explicit_dim)
        summaries = getattr(opt, f"client_{name}_features", None)
        if summaries is None or len(summaries) == 0:
            raise ValueError(f"Provide {name}_dim or load client_{name}_features before graph initialization")
        shape = summaries[0].shape
        if len(shape) != 2:
            raise ValueError(f"Client {name} summaries must be matrices")
        return int(shape[1])

    def network_config(self):
        """Constructor settings needed alongside a saved state_dict."""
        return {
            "num_clients": self.num_clients,
            "temporal_window": self.temporal_window,
            "attention_temperature": self.temperature,
            "graph_epsilon": self.epsilon,
            "gat_hidden_dim": self.hidden_dim,
            "gat_embedding_dim": self.embedding_dim,
            "gat_heads": self.n_heads,
            "gat_dropout": self.dropout,
            "spatial_dim": self.spatial_dim,
            "temporal_dim": self.temporal_dim,
        }

    def get_extra_state(self):
        # CPU tensors keep checkpoints portable and compatible with weights_only.
        return {
            "network_config": self.network_config(),
            "temporal_history": [item.detach().cpu().clone() for item in self.temporal_history],
        }

    def set_extra_state(self, state):
        if state["network_config"] != self.network_config():
            raise ValueError("Attention checkpoint configuration does not match this network")
        history = []
        for item in state["temporal_history"]:
            history.append(
                self._validate_client_summaries(
                    item,
                    "DCE temporal history",
                    self.temporal_dim,
                )
                .detach()
                .cpu()
                .clone()
            )
        self.temporal_history = history

    def _validate_client_summaries(self, values, name, feature_dim):
        if values is None:
            raise RuntimeError(f"{name} are required")
        reference = next(self.parameters())
        matrix = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
        expected = (self.num_clients, feature_dim)
        if tuple(matrix.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(matrix.shape)}")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        return matrix

    @staticmethod
    def _standardize(matrix):
        return (matrix - matrix.mean(dim=0, keepdim=True)) / matrix.std(
            dim=0,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-6)

    def _windowed_temporal_summaries(self, current, task_id):
        if (
            not isinstance(task_id, (int, np.integer))
            or isinstance(task_id, bool)
            or not 0 <= task_id <= len(self.temporal_history)
        ):
            raise ValueError(
                "BreastG-FCL graph tasks must be generated sequentially; "
                f"received task {task_id} with {len(self.temporal_history)} tasks stored"
            )
        start = max(0, task_id - self.temporal_window + 1)
        previous = [item.to(current) for item in self.temporal_history[start:task_id]]
        # The current tensor keeps its gradient; historical summaries are detached.
        return torch.cat(previous + [current], dim=1)

    def forward(self, spatial_features, temporal_features=None, task_id=0):
        spatial = self._validate_client_summaries(
            spatial_features,
            "Spatial morphology summaries",
            self.spatial_dim,
        )
        current = self._validate_client_summaries(
            temporal_features,
            "DCE temporal summaries",
            self.temporal_dim,
        )
        window = self._windowed_temporal_summaries(current, task_id)
        spatial_attention = self.spatial_attention(self._standardize(spatial))
        # Right-align recent tasks. Zero left-padding lets early tasks share the
        # same Q/K parameters as full windows, without using any future task.
        temporal_input = F.pad(
            self._standardize(window),
            (self.temporal_window * self.temporal_dim - window.shape[1], 0),
        )
        temporal_attention = self.temporal_attention(temporal_input)
        product = spatial_attention * temporal_attention
        denominator = product.sum(dim=1, keepdim=True) + self.epsilon
        graph = product / denominator.clamp_min(torch.finfo(product.dtype).tiny)
        if not torch.isfinite(graph).all():
            raise ValueError("Attention network produced non-finite graph values")

        # Commit state only after both inputs and graph calculation succeed.
        saved = current.detach().cpu().clone()
        if task_id == len(self.temporal_history):
            self.temporal_history.append(saved)
        else:
            self.temporal_history[task_id] = saved
        self.last_spatial_attention = spatial_attention.detach().cpu().numpy().copy()
        self.last_temporal_patterns = temporal_attention.detach().cpu().numpy().copy()
        self.last_temporal_window = window.detach().cpu().numpy().copy()
        return graph

    def learn(self, epochs, model_updates=None, task_id=None, spatial_features=None, temporal_features=None):
        """Generate a deterministic graph; ``epochs`` does not train attention."""
        del epochs, model_updates
        training_modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            with torch.no_grad():
                graph = self(spatial_features, temporal_features, task_id=0 if task_id is None else task_id)
            return graph.cpu().numpy().copy()
        finally:
            for module, training in training_modes:
                module.training = training
