from math import gcd
from typing import Dict

import torch
import torch.nn as nn
from torch.nn import functional as F


class GraphAttention(nn.Module):
    """
    Graph attention module to pool features from source nodes to destination nodes.

    Given a destination node i, we aggregate the features from nearby source nodes j whose L2
    distance from the destination node i is smaller than a threshold.

    This graph attention module follows the implementation in LaneGCN and is slightly different
    from the one in Gragh Attention Networks.

    Compared to the open-sourced LaneGCN, this implementation omitted a few LayerNorm operations
    after some layers. Not sure if they are needed or not.
    """

    def __init__(self, src_feature_len: int, dst_feature_len: int, dist_threshold: float):
        super().__init__()
        """
        :param src_feature_len: source node feature length.
        :param dst_feature_len: destination node feature length.
        :param dist_threshold:
            Distance threshold in meters.
            We only aggregate node information if the destination nodes are within this distance
            threshold from the source nodes.
        """
        self.dist_threshold = dist_threshold

        self.src_encoder = nn.Sequential(
            nn.Linear(src_feature_len, src_feature_len),
            nn.ReLU(inplace=True),
        )

        self.dst_encoder = nn.Sequential(
            nn.Linear(dst_feature_len, dst_feature_len),
            nn.ReLU(inplace=True),
        )

        # Just use the destination node feature length as the edge distance feature length.
        edge_dist_feature_len = dst_feature_len
        self.edge_dist_encoder = nn.Sequential(
            nn.Linear(2, edge_dist_feature_len),
            nn.ReLU(inplace=True),
        )

        edge_input_feature_len = src_feature_len + edge_dist_feature_len + dst_feature_len
        edge_output_feature_len = dst_feature_len
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_feature_len, edge_output_feature_len),
            nn.ReLU(inplace=True),
            nn.Linear(edge_output_feature_len, edge_output_feature_len),
        )

        self.dst_feature_norm = nn.LayerNorm(dst_feature_len)

        self.output_linear = nn.Linear(dst_feature_len, dst_feature_len)

    def forward(
        self,
        src_node_features: torch.Tensor,
        src_node_pos: torch.Tensor,
        dst_node_features: torch.Tensor,
        dst_node_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Graph attention module to pool features from source nodes to destination nodes.

        :param src_node_features: <torch.FloatTensor: num_src_nodes, src_node_feature_len>.
            Source node features.
        :param src_node_pos: <torch.FloatTensor: num_src_nodes, 2>. Source node (x, y) positions.
        :param dst_node_features: <torch.FloatTensor: num_dst_nodes, dst_node_feature_len>.
            Destination node features.
        :param dst_node_pos: <torch.FloatTensor: num_dst_nodes, 2>. Destination node (x, y)
            positions.
        :return: <torch.FloatTensor: num_dst_nodes, dst_node_feature_len>. Output destination
            node features.
        """
        # Find (src, dst) node pairs that are within the distance threshold,
        # and they form the edges of the graph.
        # src_dst_dist.shape is (num_src_nodes, num_dst_nodes).
        src_dst_dist = (src_node_pos.view(-1, 1, 2) - dst_node_pos.view(1, -1, 2)).norm(dim=-1)
        src_dst_dist_mask = src_dst_dist <= self.dist_threshold
        # edge_src_dist_pairs.shape is (num_edges, 2).
        edge_src_dist_pairs = src_dst_dist_mask.nonzero(as_tuple=False)
        edge_src_idx = edge_src_dist_pairs[:, 0]
        edge_dst_idx = edge_src_dist_pairs[:, 1]

        src_node_encoded_features = self.src_encoder(src_node_features)
        dst_node_encoded_features = self.dst_encoder(dst_node_features)

        # edge_src_features.shape is (num_edges, edge_src_feature_len).
        edge_src_features = src_node_encoded_features[edge_src_idx]
        edge_dst_features = dst_node_encoded_features[edge_dst_idx]
        # edge_src_pos.shape is (num_edges, 2).
        edge_src_pos = src_node_pos[edge_src_idx]
        edge_dst_pos = dst_node_pos[edge_dst_idx]
        edge_dist = self.edge_dist_encoder(edge_src_pos - edge_dst_pos)
        edge_input_features = torch.cat([edge_src_features, edge_dist, edge_dst_features], dim=-1)
        edge_output_features = self.edge_encoder(edge_input_features)

        # Aggregate the edge features.
        dst_node_output_features = dst_node_encoded_features.clone()
        dst_node_output_features.index_add_(0, edge_dst_idx, edge_output_features)

        # Normalize the feature dimension after adding the edge features and apply activation.
        dst_node_output_features = self.dst_feature_norm(dst_node_output_features)
        dst_node_output_features = F.relu(dst_node_output_features, inplace=True)

        # One more linear layer.
        dst_node_output_features = self.output_linear(dst_node_output_features)

        # Add residual connection and do a final activation at the end.
        dst_node_output_features += dst_node_features
        dst_node_output_features = F.relu(dst_node_output_features, inplace=True)

        return dst_node_output_features


class Lane2ActorAttention(nn.Module):
    """
    Lane-to-Actor attention module.
    """

    def __init__(self, lane_feature_len: int, actor_feature_len: int, num_attention_layers: int,
                 dist_threshold_m: float) -> None:
        """
        :param lane_feature_len: Lane feature length.
        :param actor_feature_len: Actor feature length.
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param dist_threshold_m:
            Distance threshold in meters.
            We only aggregate map-to-actor node
            information if the actor nodes are within this distance threshold from the lane nodes.
            The value used in the LaneGCN paper is 100 meters.
        """
        super().__init__()
        attention_layers = [
            GraphAttention(lane_feature_len, actor_feature_len, dist_threshold_m)
            for _ in range(num_attention_layers)
        ]
        self.attention_layers = nn.ModuleList(attention_layers)

    def forward(
        self,
        lane_features: torch.Tensor,
        lane_centers: torch.Tensor,
        actor_features: torch.Tensor,
        actor_centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform Lane-to-Actor attention.

        :param actor_features: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features.
        :param actor_centers: <torch.FloatTensor: num_actors, 2>. (x, y) positions of the actors.
        :param lane_features: <torch.FloatTensor: num_lanes, lane_feature_len>. Lane features.
            Features corresponding to map nodes.
        :param lane_centers: <torch.FloatTensor: num_lanes, 2>. (x, y) positions of the lanes.
        :return: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features after
            aggregating the lane features.
        """
        for attention_layer in self.attention_layers:
            actor_features = attention_layer(
                lane_features,
                lane_centers,
                actor_features,
                actor_centers,
            )
        return actor_features


class Actor2ActorAttention(nn.Module):
    """
    Actor-to-Actor attention module.
    """

    def __init__(self, actor_feature_len: int, num_attention_layers: int, dist_threshold_m: float) -> None:
        """
        :param actor_feature_len: Actor feature length.
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param dist_threshold_m:
            Distance threshold in meters.
            We only aggregate actor-to-actor node
            information if the actor nodes are within this distance threshold from the other actor nodes.
            The value used in the LaneGCN paper is 30 meters.
        """
        super().__init__()
        attention_layers = [
            GraphAttention(actor_feature_len, actor_feature_len, dist_threshold_m)
            for _ in range(num_attention_layers)
        ]
        self.attention_layers = nn.ModuleList(attention_layers)

    def forward(
        self,
        actor_features: torch.Tensor,
        actor_centers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform Actor-to-Actor attention.

        :param actor_features: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features.
        :param actor_centers: <torch.FloatTensor: num_actors, 2>. (x, y) positions of the actors.
        :return: <torch.FloatTensor: num_actors, actor_feature_len>. Actor features after aggregating the lane features.
        """
        for attention_layer in self.attention_layers:
            actor_features = attention_layer(
                actor_features,
                actor_centers,
                actor_features,
                actor_centers,
            )
        return actor_features


class LaneNet(nn.Module):
    """
    Lane feature extractor with either lane graph convolution or MLP
    """

    def __init__(self, lane_input_len: int, lane_feature_len: int, num_scales: int,
                 num_res_blocks: int, is_map_feat: bool) -> None:
        """
        Constructs LaneGraphCNN layer for LaneGCN.
        :param lane_input_len: Raw feature size of lane vector representation (e.g. 2 if using
            average of x,y coordinates of lane end points)
        :param lane_feature_len: Feature size of lane nodes.
        :param num_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param is_map_feature: if set to True, output max pooling over the lane features so it can
            be used as a map feature, otherwise output lane features as is.
        """
        super().__init__()
        self.is_map_feat = is_map_feat
        num_groups = 1
        self.num_scales = num_scales

        self.input = nn.Sequential(
            nn.Linear(lane_input_len, lane_feature_len),
            nn.ReLU(inplace=True),
            LinearWithGroupNorm(lane_feature_len, lane_feature_len, num_groups=num_groups, activation=False),
        )

        self.seg = nn.Sequential(
            nn.Linear(lane_input_len, lane_feature_len),
            nn.ReLU(inplace=True),
            LinearWithGroupNorm(lane_feature_len, lane_feature_len, num_groups=num_groups, activation=False),
        )

        self._relu = nn.ReLU(inplace=True)

    def forward(self, vector_map: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """
        :param vector_map:
            <Dict[str, torch.FloatTensor]>
            Vector map inputs as a dict of {
                'coords':
                    <torch.FloatTensor: num_lanes, 2, 2>. Coordindates of the start and
                    end point of each lane segment.
                'connections':
                    <torch.LongTensor: num_connections, 2>. Indices of the predecessor
                    and successor segment pair.
            }
        :return:
            lane_features: <torch.FloatTensor: num lane segments across all batches,
               map feature size>. Features corresponding to lane nodes, updated with
               information from adjacent lane nodes.
        """
        lane_centers = vector_map['coords'].mean(axis=1)  # mean of start and end point
        lane_diff = vector_map['coords'][:, 1] - vector_map['coords'][:, 0]
        lane_features = self.input(lane_centers)
        lane_features += self.seg(lane_diff)
        lane_features = self._relu(lane_features)  # Generating x_i for all lane nodes

        if self.is_map_feat:
            return torch.max(lane_features, 0, keepdim=True)[0]
        else:
            return lane_features


class Lane2Lane(nn.Module):
    """
    The lane to lane block ropagates information over lane graphs and updates the lane feature
    """

    def __init__(self, lane_feature_len: int, num_scales: int, num_res_blocks: int) -> None:
        """
        Constructs Fusion Net among lane nodes.
        :param lane_feature_len: Feature size of lane nodes.
        :param num_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        """
        super().__init__()
        num_groups = 1

        fusion_components = ['ctr', 'norm', 'ctr2']
        for i in range(num_scales):
            fusion_components.append('pre' + str(i))
            fusion_components.append('suc' + str(i))

        fusion_net: Dict[str, nn.ModuleList] = dict()
        for key in fusion_components:
            fusion_net[key] = []

        for i in range(num_res_blocks):  # set to 4 in LaneGCN
            for key in fusion_net:
                if key in ['norm']:
                    fusion_net[key].append(nn.GroupNorm(gcd(num_groups, lane_feature_len),
                                                        lane_feature_len))
                elif key in ['ctr2']:
                    fusion_net[key].append(LinearWithGroupNorm(lane_feature_len, lane_feature_len,
                                                               num_groups=num_groups, activation=False))
                else:
                    fusion_net[key].append(nn.Linear(lane_feature_len, lane_feature_len,
                                                     bias=False))

        for key in fusion_net:
            fusion_net[key] = nn.ModuleList(fusion_net[key])
        self.fusion_net = nn.ModuleDict(fusion_net)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, lane_features: torch.FloatTensor,
                lane_graph: Dict[str, Dict[str, torch.Tensor]]) -> torch.FloatTensor:
        """
        :param lane_features: <torch.FloatTensor: num lane nodes across all batches,
            lane node feature size>. Features corresponding to lane nodes.
        :param lane_graph: <Dict[str, List[torch.Tensor]]: Extracted lane graph from MapNet()>
            n_hop_pre: List of n_hop pre neighbor node index, torch.Tensor: num of lane nodes
            suc: List of cooresponding successor nodes, torch.Tensor: num of lane nodes
            n_hop_suc: List of n_hop suc neighbor node index, torch.Tensor: num of lane nodes
            pre: List of cooresponding precessor nodes, torch.Tensor: num of lane nodes
        :return: lane_features: <torch.FloatTensor: num lane segments across all batches,
                                map feature size>.
            Features corresponding to lane nodes, updated with information from adjacent
                lane nodes.
        """

        res = lane_features
        for i in range(len(self.fusion_net['ctr'])):
            temp = self.fusion_net['ctr'][i](lane_features)
            for key in self.fusion_net:
                if key.startswith('pre'):
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        lane_graph['suc'][str(k2)],
                        self.fusion_net[key][i](lane_features[
                            lane_graph['n_hop_pre'][str(k2)]
                        ]),
                    )

                if key.startswith('suc'):
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        lane_graph['pre'][str(k2)],
                        self.fusion_net[key][i](lane_features[
                            lane_graph['n_hop_suc'][str(k2)]
                        ]),
                    )

            lane_features = self.fusion_net['norm'][i](temp)
            lane_features = self._relu(lane_features)

            lane_features = self.fusion_net['ctr2'][i](lane_features)
            lane_features += res
            lane_features = self._relu(lane_features)
            res = lane_features

        return lane_features


class LinearWithGroupNorm(nn.Module):
    def __init__(self, n_in: int, n_out: int, num_groups: int = 32,
                 activation: bool = True) -> None:
        """
        Linear layer used in LaneGCN.
        :param n_in: Number of input channels.
        :param n_out: Number of output channels.
        :param num_groups: Number of groups for GroupNorm.
        :param activation: Boolean indicating whether to apply ReLU activation.
        """
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.norm = nn.GroupNorm(gcd(num_groups, n_out), n_out)
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear layer to input tensor.
        :param x: Input tensor.
        :return: Output of linear layer.
        """
        out = self.linear(x)
        out = self.norm(out)
        if self.activation:
            out = self.relu(out)
        return out
