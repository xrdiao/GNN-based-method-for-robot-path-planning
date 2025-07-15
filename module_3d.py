import torch
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, ReLU, Conv3d
import numpy as np
from torch_geometric.loader import DataLoader
import networkx as nx
from my_graph import my_graph
from maps import maps


class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channel=21, encode_size=128, cuda=False, **kwargs):
        super(NodeEncoder, self).__init__()
        self.conv1 = SAGEConv(in_channel, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, encode_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x


class MapEncoder(torch.nn.Module):
    def __init__(self, in_channel, encode_size):
        super(MapEncoder, self).__init__()
        self.net = torch.nn.Sequential(Conv3d(in_channel, 256, kernel_size=(3, 5, 5)),
                                       ReLU(),
                                       Conv3d(256, 128, kernel_size=(3, 3, 3)),
                                       ReLU())
        self.lin = Linear(114048, encode_size)
        torch.nn.init.xavier_normal_(self.lin.weight)

    def forward(self, data):
        x = data
        x = self.net(x)
        x = self.lin(torch.flatten(x))
        return x


class MyGnn(torch.nn.Module):
    def __init__(self, in_channel, out_channel, batch, encode_size=64, cuda=True, **kwargs):
        super(MyGnn, self).__init__()
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.node_encoder = NodeEncoder(in_channel, encode_size, cuda, **kwargs)
        self.map_encoder = MapEncoder(1, encode_size)

        self.lin1 = Linear(encode_size, 64)
        self.lin2 = Linear(64, out_channel)
        self.a = 0.35
        self.b = 1 - self.a

    def forward(self, node_data, edge_index, map_data):
        node_data = node_data.to(self.device)
        edge_index = edge_index.to(self.device)
        map_data = map_data.to(self.device)

        node_encoder = self.node_encoder(node_data, edge_index)
        map_encoder = self.map_encoder(map_data)
        x = self.a * self.norm_data(node_encoder) + self.b * self.norm_data(map_encoder)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.softmax(x, dim=1)

    def norm_data(self, x):
        min_x = torch.min(x)
        max_x = torch.max(x)

        x = (x - min_x) / (max_x - min_x)
        return x

    def data2label(self, data):
        a = torch.argmax(data, dim=1)
        label = np.zeros([200, 3])
        for i in range(200):
            label[i, a[i]] = 1
        return label
