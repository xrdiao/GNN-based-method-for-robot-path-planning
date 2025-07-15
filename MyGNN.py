import torch
from torch_geometric.nn import SAGEConv, GATConv
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Conv3d
import numpy as np
from map_3d import maps_3d


class NodeEncoder(torch.nn.Module):
    def __init__(self, in_channel=1, encode_size=128, cuda=False, **kwargs):
        super(NodeEncoder, self).__init__()
        self.conv1 = GATConv(in_channel, 256)
        self.conv2 = SAGEConv(256, 128)
        self.conv3 = SAGEConv(128, encode_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        node_coder = self.conv3(x, edge_index)
        return node_coder


class MapEncoder(torch.nn.Module):
    def __init__(self, map_chanel, obs_channel, encode_size):
        super(MapEncoder, self).__init__()
        self.net1 = torch.nn.Sequential(Conv3d(map_chanel, 256, kernel_size=(3, 5, 5)),
                                        ReLU(),
                                        Conv3d(256, 128, kernel_size=(3, 3, 3)),
                                        ReLU())

        self.lin1 = Linear(114048, encode_size)
        torch.nn.init.xavier_normal_(self.lin1.weight)

    def forward(self, raw_map):
        map_coder = raw_map
        map_coder = self.net1(map_coder)

        # 如果有办法把变长的数据当初处理成定长的数组，那么可以考虑将obs加进来
        env_coder = self.lin1(torch.flatten(map_coder))
        return env_coder


class EnvCoder(torch.nn.Module):
    def __init__(self, node_channel, map_channel, obs_channel, out_channel, batch, encode_size=64, cuda=False,
                 **kwargs):
        super(EnvCoder, self).__init__()

        self.node_coder = NodeEncoder(node_channel, encode_size, cuda, **kwargs)
        self.map_coder = MapEncoder(map_channel, obs_channel, encode_size)

        self.lin1 = Linear(encode_size, 64)
        self.lin2 = Linear(64, out_channel)
        self.a = 0.35
        self.b = 1 - self.a

    def forward(self, node_data, edge_index, map_data):
        node_coder = self.node_coder(node_data, edge_index)
        map_coder = self.map_coder(raw_map=map_data)
        x = self.a * self.norm_data(node_coder) + self.b * self.norm_data(map_coder)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def norm_data(self, x):
        min_x = torch.min(x)
        max_x = torch.max(x)
        x = (x - min_x) / (max_x - min_x)
        return x


class PathEncoder(torch.nn.Module):
    '''
    本质是一个RNN，由于调库的时候反向传播有问题，所以复现了一下
    '''

    def __init__(self, input_size=7, hidden_size=32, num_layers=1, h_state=None, device=torch.device('cpu')):
        # hidden_size就是输出的维度
        super(PathEncoder, self).__init__()
        self.relu = ReLU()
        self.W = Linear(7, 32)
        self.U = Linear(32, 32)
        self.bias1 = torch.empty(1, 32).to(device)
        self.bias2 = torch.empty(1, 32).to(device)
        torch.nn.init.normal_(self.bias1, mean=0.0, std=0.5)
        torch.nn.init.normal_(self.bias2, mean=0.0, std=0.5)
        self.h_state = h_state

    def forward(self, data):
        Hidden = None
        Out = None
        for path in data:
            Hidden = self.relu(self.W(path) + self.bias1)
            Out = self.U(Hidden) + self.bias2
        self.h_state = Hidden
        return Out.squeeze()


class Attention(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, embedding_size, dk, size):
        super(Attention, self).__init__()
        self.key = Linear(key_size, embedding_size)
        self.query = Linear(query_size, embedding_size)
        self.query_env = Linear(query_size, embedding_size)
        self.value = Linear(value_size, embedding_size)
        self.Lin = Linear(size + 1, embedding_size)
        self.dk = dk
        self.layer_norm = torch.nn.LayerNorm(embedding_size, eps=1e-6)

    def forward(self, env, path):
        # 根据transformer的运行逻辑写的
        env_key = self.key(env)

        path_value = self.value(path)
        env_value = self.value(env)

        env_query = self.query(env)
        path_query = self.query(path)

        self_attention = torch.matmul(env_query, env_key.T)
        path_attention = torch.matmul(path_query, env_key.T).unsqueeze(dim=-1)
        env_attention = torch.cat((self_attention, path_attention), dim=-1)
        env_attention = (env_attention / self.dk).softmax(dim=-1)

        # env_code的维度为n*32
        env_code = (env_attention.unsqueeze(-1) * torch.cat(
            (env_value.unsqueeze(1).repeat(1, len(env), 1), path_value.unsqueeze(0).repeat(len(env), 1, 1)),
            dim=1)).sum(dim=1)

        # 残差网络并通过layer_norm实现归一化，使样本分布稳定
        return self.layer_norm(env + env_code)


class Guidance(torch.nn.Module):
    def __init__(self, node_channel, map_chanel, obs_channel, out_channel, batch=1, encode_size=32, input_size=7,
                 hidden_size=32, num_layers=1, h_state=None, cuda=True, nodes_size=200, init_net=False, **kwargs):
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        super(Guidance, self).__init__()
        self.env_encoder = EnvCoder(node_channel, map_chanel, obs_channel, out_channel, batch, encode_size=encode_size,
                                    cuda=cuda, **kwargs)
        self.path_encoder = PathEncoder(input_size, hidden_size, num_layers, h_state, self.device)
        self.attention = Attention(hidden_size, out_channel, out_channel, encode_size, np.sqrt(encode_size), nodes_size)

        self.decode = torch.nn.Sequential(Linear(encode_size, encode_size),
                                          ReLU(),
                                          Linear(encode_size, encode_size))
        self.guidance = torch.nn.Sequential(Linear(3 * encode_size, encode_size),
                                            ReLU(),
                                            Linear(encode_size, 1))
        self.env_coder = None

        if init_net:
            for k in self.modules():
                if isinstance(k, torch.nn.Conv3d):
                    print('初始化Conv层参数')
                    torch.nn.init.kaiming_uniform_(k.weight.data, mode='fan_in', nonlinearity='relu')
                elif isinstance(k, Linear):
                    print('初始化Linear层参数')
                    torch.nn.init.kaiming_uniform_(k.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, node_raw, edge_index, map_raw, path):
        node_raw = node_raw.to(self.device)
        edge_index = edge_index.to(self.device)
        map_raw = map_raw.to(self.device)
        path = path.to(self.device)

        torch.backends.cudnn.enabled = False
        if self.env_coder is None:
            self.env_coder = self.env_encoder(node_data=node_raw, edge_index=edge_index, map_data=map_raw)
            # print('new map')
        path_coder = self.path_encoder(path)

        encoder = self.attention(self.env_coder, path_coder)
        decoder = self.decode(encoder)

        guidance = torch.cat((decoder[edge_index[0, :]], decoder[edge_index[0, :]] - decoder[edge_index[1, :]],
                              self.env_coder[edge_index[1, :]]), dim=-1)

        guidance = self.guidance(guidance)
        guidance = torch.abs(guidance)
        output = torch.zeros([len(node_raw), len(node_raw)]).to(self.device)
        output[edge_index[0, :], edge_index[1, :]] = guidance.squeeze()
        return output


if __name__ == '__main__':
    m = maps_3d()

    train = torch.load("data/train.pth")
    maps = torch.load("data/map.pth")

    m.init_problem(0)

    g = Guidance(node_channel=21, map_chanel=1, obs_channel=6, out_channel=32, batch=1)

    i = 0
    for data, env in zip(train, maps):
        # obs_raw = torch.tensor(torch.reshape(torch.tensor(np.array(m.problems[i][0])), [-1, 6]), dtype=torch.float)
        out = g(data, env, m.path)
        i += 1
        print(out.shape)

    # print(out)
