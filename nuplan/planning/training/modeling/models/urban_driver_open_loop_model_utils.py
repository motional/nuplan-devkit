"""
Copyright 2022 Motional

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


def pad_avails(avails: torch.Tensor, pad_to: int, dim: int) -> torch.Tensor:
    """
    Copied from L5Kit's implementation `pad_avail`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/common.py.
    Changes:
        1. Change function name `pad_avail` to `pad_avails`
        2. Add dimension checking and adjust output dimension

    Pad vectors to 'pad_to' size. Dimensions are:
    N: number of elements (polylines)
    P: number of points
    :param avails: Availabilities to be padded, should be (N,P) and we're padding dim.
    :param pad_to: Number of elements or points.
    :param dim: Dimension at which to apply padding.
    :return: The padded polyline availabilities (N,P).
    """
    num_els, num_points = avails.shape
    if dim == 0 or dim == -2:
        num_els = pad_to - num_els
    elif dim == 1 or dim == -1:
        num_points = pad_to - num_points
    else:
        raise ValueError(dim)
    pad = torch.zeros(num_els, num_points, dtype=avails.dtype, device=avails.device)
    return torch.cat([avails, pad], dim=dim)


def pad_polylines(polylines: torch.Tensor, pad_to: int, dim: int) -> torch.Tensor:
    """
    Copied from L5Kit's implementation `pad_points`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/common.py.
    Changes:
        1. Change function name `pad_points` to `pad_polylines`
        2. Add dimension checking and adjust output dimension

    Pad vectors to 'pad_to' size. Dimensions are:
    N: number of elements (polylines)
    P: number of points
    F: number of features
    :param polylines: Polylines to be padded, should be (N,P,F) and we're padding dim.
    :param pad_to: Number of elements, points, or features.
    :param dim: Dimension at which to apply padding.
    :return: The padded polylines (N,P,F).
    """
    num_els, num_points, num_feats = polylines.shape
    if dim == 0 or dim == -3:
        num_els = pad_to - num_els
    elif dim == 1 or dim == -2:
        num_points = pad_to - num_points
    elif dim == 2 or dim == -1:
        num_feats = pad_to - num_feats
    else:
        raise ValueError(dim)
    pad = torch.zeros(num_els, num_points, num_feats, dtype=polylines.dtype, device=polylines.device)
    return torch.cat([polylines, pad], dim=dim)


class LocalMLP(nn.Module):
    """
    A Local 1-layer MLP.
    Copied from L5Kit's implementation `LocalMLP`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/local_graph.py.
    Changes:
        1. Change input & output description
    """

    def __init__(self, dim_in: int, use_norm: bool = True):
        """
        Constructs LocalMLP.
        :param dim_in: Input feature size.
        :param use_norm: Whether to apply layer norm, defaults to True.
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward of the module.
        :param x: Input tensor (..., dim_in).
        :return: Output tensor (..., dim_in).
        """
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class MLP(nn.Module):
    """
    Copied from L5Kit's implementation `MLP`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/global_graph.py.
    Changes:
        1. Add input & output description for `__init__`, `reset_parameters`, `forward`
        2. Change variable name `h` to `hidden_dims` in `__init__`
        3. Change variable name `i` to `layer_idx` in `forward`

    Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        """
        Constructs MLP.
        :param input_dim: Input feature size.
        :param hidden_dim: Hidden layer size.
        :paran output_dim: Output feature size.
        :param num_layers: Number of model layers.
        """
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Re-initialize layer parameters.
        """
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward of the module.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for layer_idx, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if layer_idx < self.num_layers - 1 else layer(x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Copied from L5Kit's implementation `SinusoidalPositionalEmbedding`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/local_graph.py.
    Changes:
        1. Change input variable name `d_model` to `embedding_size`
        2. Change variable name `pe` to `pos_encoding`
        3. Change variable name `t` to `seq_idx`

    A positional embedding module.
    Useful to inject the position of sequence elements in local graphs.
    """

    def __init__(self, embedding_size: int, max_len: int = 5000):
        """
        Constructs positional embedding module.
        :param embedding_size: Feature size.
        :param max_len: Max length of the sequences, defaults to 5000.
        """
        super().__init__()
        pos_encoding = torch.zeros(max_len, embedding_size)
        seq_idx = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        log_value = torch.log(torch.tensor([1e4])).item()
        omega = torch.exp((-log_value / embedding_size) * torch.arange(0, embedding_size, 2).float())
        pos_encoding[:, 0::2] = torch.sin(seq_idx * omega)
        pos_encoding[:, 1::2] = torch.cos(seq_idx * omega)
        self.register_buffer("static_embedding", pos_encoding.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward of the module.
        :param x: Input tensor of shape batch_size x num_agents x sequence_length x d_model.
        :return: Output tensor.
        """
        return self.static_embedding[: x.shape[2], :]


class TypeEmbedding(nn.Module):
    """
    Adapted from L5Kit's implementation `VectorizedEmbedding`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/global_graph.py.
    Changes:
        1. Change input variable name `d_model` to `embedding_size`
        2. Change variable name `pe` to `pos_encoding`
        3. Change variable name `t` to `seq_idx`
        4. Change `forward` with own feature types

    A module which associates learnable embeddings to types.
    """

    def __init__(self, embedding_dim: int, feature_types: Dict[str, int]):
        """
        Constructs TypeEmbedding.
        :param embedding_dim: Feature embedding dimensionality.
        :param feature_types: Dict representing feature types keyed by name (Torchscript not supportive of enums).
        """
        super(TypeEmbedding, self).__init__()
        self._feature_types = feature_types
        self.embedding = nn.Embedding(len(self._feature_types), embedding_dim)

    def forward(
        self,
        batch_size: int,
        agents_len: int,
        agent_features: List[str],
        map_features: List[str],
        map_features_len: Dict[str, int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Forward of the module: embed the given elements based on their type.
        Assumptions:
        - agent of interest is the first one in the batch
        - other agents follow
        - then we have map features (polylines)
        :param batch_size: number of samples in batch.
        :param agents_len: number of agents.
        :param agent_features: list of agent feature types.
        :param map_features: list of map feature types.
        :param map_features_len: number of map features per type.
        :param device: desired device of tensors to supply to torch.
        :return Output tensor.
        """
        with torch.no_grad():

            total_agents_len = agents_len * len(agent_features)
            total_len = 1 + total_agents_len + sum(map_features_len.values())
            agents_start_idx = 1
            map_start_idx = agents_start_idx + total_agents_len

            indices = torch.full(
                (batch_size, total_len),
                fill_value=self._feature_types["NONE"],
                dtype=torch.long,
                device=device,
            )

            # ego
            indices[:, 0].fill_(self._feature_types["EGO"])

            # other agents
            for feature_name in agent_features:
                indices[:, agents_start_idx : agents_start_idx + agents_len].fill_(self._feature_types[feature_name])
                agents_start_idx += agents_len

            # map features
            for feature_name in map_features:
                feature_len = map_features_len[feature_name]
                indices[:, map_start_idx : map_start_idx + feature_len].fill_(self._feature_types[feature_name])
                map_start_idx += feature_len

        return self.embedding.forward(indices)


class LocalSubGraphLayer(nn.Module):
    """
    Copied from L5Kit's implementation `LocalSubGraphLayer`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/local_graph.py.
    Changes:
        1. Change input & output description
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        """
        Constructs local subgraph layer.
        :param dim_in: Input feat size.
        :param dim_out: Output feat size.
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = LocalMLP(dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward of the module.
        :param x: Input tensor [num_elements, num_points, dim_in].
        :param invalid_mask: Invalid mask for x [batch_size, num_elements, num_points].
        :return: Output tensor [num_elements, num_points, dim_out].
        """
        # x input -> num_elements * num_points * embedded_vector_length
        _, num_points, _ = x.shape
        # x mlp -> num_elements * num_points * dim_in
        x = self.mlp(x)
        # compute the masked max for each feature in the sequence
        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float("-inf"))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        # repeat it along the sequence length
        x_agg = x_agg.repeat(1, num_points, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)  # remap to a possibly different feature length
        return x


class LocalSubGraph(nn.Module):
    """
    Copied from L5Kit's implementation `LocalSubGraph`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/local_graph.py.
    Changes:
        1. Change input & output description

    PointNet-like local subgraph - implemented as a collection of local graph layers.
    """

    def __init__(self, num_layers: int, dim_in: int) -> None:
        """
        :param num_layers: Number of LocalSubGraphLayers.
        :param dim_in: Input, hidden, output dim for features.
        """
        super(LocalSubGraph, self).__init__()
        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        for _ in range(num_layers):
            self.layers.append(LocalSubGraphLayer(dim_in, dim_in))

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """
        Forward of the module.
        - Add positional encoding
        - Forward to layers
        - Aggregates using max
        (calculates a feature descriptor per element - reduces over points)
        :param x: Input tensor [batch_size, num_elements, num_points, dim_in].
        :param invalid_mask: Invalid mask for x [batch_size, num_elements, num_points].
        :param pos_enc: Positional_encoding for x.
        :return: Output tensor [batch_size, num_elements, num_points, dim_in].
        """
        batch_size, num_elements, num_points, dim_in = x.shape

        x += pos_enc
        # exclude completely invalid sequences from local subgraph to avoid NaN in weights
        x_flat = x.view(-1, num_points, dim_in)
        invalid_mask_flat = invalid_mask.view(-1, num_points)
        # (batch_size x (1 + M),)
        valid_polys = ~invalid_mask.all(-1).flatten()
        # valid_seq x seq_len x vector_size
        x_to_process = x_flat[valid_polys]
        mask_to_process = invalid_mask_flat[valid_polys]
        for layer in self.layers:
            x_to_process = layer(x_to_process, mask_to_process)

        # aggregate sequence features
        x_to_process = x_to_process.masked_fill(mask_to_process[..., None] > 0, float("-inf"))
        # valid_seq x vector_size
        x_to_process = torch.max(x_to_process, dim=1).values

        # restore back the batch
        x = torch.zeros_like(x_flat[:, 0])
        x[valid_polys] = x_to_process
        x = x.view(batch_size, num_elements, self.dim_in)
        return x


class MultiheadAttentionGlobalHead(nn.Module):
    """
    Copied from L5Kit's implementation `MultiheadAttentionGlobalHead`:
    https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/global_graph.py.
    Changes:
        1. Add input & output description for `__init__`, `forward`
        2. Add num_mlp_layers & hidden_size_scaling to adjust MLP layers
        3. Change input variable `d_model` to `global_embedding_size`

    Global graph making use of multi-head attention.
    """

    def __init__(
        self,
        global_embedding_size: int,
        num_timesteps: int,
        num_outputs: int,
        nhead: int = 8,
        dropout: float = 0.1,
        hidden_size_scaling: int = 4,
        num_mlp_layers: int = 3,
    ):
        """
        Constructs global multi-head attention layer.
        :param global_embedding_size: Feature size.
        :param num_timesteps: Number of output timesteps.
        :param num_outputs: Number of output features per timestep.
        :param nhead: Number of attention heads. Default 8: query=ego, keys=types,ego,agents,map, values=ego,agents,map.
        :param dropout: Float in range [0,1] for level of dropout. Set to 0 to disable it. Default 0.1.
        :param hidden_size_scaling: Controls hidden layer size, scales embedding dimensionality. Default 4.
        :param num_mlp_layers: Num MLP layers. Default 3.
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_outputs = num_outputs
        self.encoder = nn.MultiheadAttention(global_embedding_size, nhead, dropout=dropout)
        self.output_embed = MLP(
            global_embedding_size,
            global_embedding_size * hidden_size_scaling,
            num_timesteps * num_outputs,
            num_mlp_layers,
        )

    def forward(
        self, inputs: torch.Tensor, type_embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward of the module.
        :param inputs: Model inputs. [1 + N + M, batch_size, feature_dim]
        :param type_embedding: Type embedding describing the different input types. [1 + N + M, batch_size, feature_dim]
        :param mask: Availability mask. [batch_size, 1 + N + M]
        :return Tuple of outputs, attention.
        """
        # dot-product attention:
        #   - query is ego's vector
        #   - key is inputs plus type embedding
        #   - value is inputs
        out, attns = self.encoder(inputs[[0]], inputs + type_embedding, inputs, mask)
        outputs = self.output_embed(out[0]).view(-1, self.num_timesteps, self.num_outputs)
        return outputs, attns
