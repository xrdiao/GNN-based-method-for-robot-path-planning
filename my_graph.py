import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from maps import maps
import time
from Dijkstra import Dijkstra
from torch.utils.data.sampler import WeightedRandomSampler

LIMITE = 1.
inf = 10000


class my_graph():
    '''
        这个类主要用来处理生成的随机图向torch类型的转变，说白了就是为图神经网络构建数据集用的。
        它的内容包含：将邻接矩阵转成邻接列表，给节点贴标签，获得节点特征等。
        贴标签，原理同对edge贴标签，和edge的构建一样，搞一个优先级，按规划器给出的路径，与路径邻近的节点，其他节点来进行分级
        节点的特征构成如下[x, y, x_init, y_init, x_goal, y_goal, degree{}]
        edge由邻接矩阵获得
        y=[1,0,0],[0,1,0],[0,0,1]分别对应不同等级
        在最后规划的时候，优先连接高优先级的节点，如果失败，则将下一优先级的节点加入图中
    '''

    def __init__(self, maps):
        self.map = maps
        self.nodes = maps.nodes
        self.feature = []
        self.edge = None
        self.label_nodes = []
        self.label_edges = []

    def reset_para(self):
        self.nodes = self.map.nodes
        self.feature = []
        self.edge = None
        self.label_nodes = []
        self.label_edges = []

    def matrix2list(self, matrix=None):
        '''

        Args:
            matrix: adjacent matrix

        Returns:
            adjacent list

        '''
        assert matrix is not None
        edge_index = np.array(np.where(matrix < 20))
        edge_index = [np.hstack([edge_index[0], edge_index[1]]), np.hstack([edge_index[1], edge_index[0]])]
        # 无向图
        self.edge = torch.tensor(np.array(edge_index), dtype=torch.int64)
        return self.edge

    def get_feature(self):
        for i in range(len(self.nodes)):
            coor = self.nodes[i]
            self.feature.append(np.hstack([coor, self.map.goal_state]))
        feature = torch.tensor(np.array(self.feature), dtype=torch.float)
        self.feature = []
        return feature

    def get_label(self):
        '''
            # 这里还需要考虑负样本，所以label将会升成4维，最后一维表示障碍物中的点
        '''
        label_node = []
        near_path = np.where(self.map.graph[self.map.path] < LIMITE * 5)[1]
        near_path = np.unique(near_path)

        for i in range(len(self.map.nodes)):
            if i in self.map.path:
                label_node.append([1, 0, 0])  # 路径上的点
            elif i in near_path:
                label_node.append([0, 1, 0])  # 路径旁的点
            else:
                label_node.append([0, 0, 1])  # 其余点
        label_node = torch.tensor(label_node, dtype=torch.float)
        self.label_nodes.append(label_node)
        return label_node

    def get_data(self):
        x_feature = self.get_feature()
        edge = self.matrix2list(self.map.graph)
        data = Data(x=x_feature, edge_index=edge)
        return data

    def get_loader(self, train_data, batch_size=1):
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
        return data_loader

    def save_label(self):
        np.save('nodes_label', self.label_nodes)


if __name__ == '__main__':
    map = maps()
    graph = my_graph(map)
    map.init_problem(0)

    n = 200
    nodes, _ = map.sample_points(n)
    nodes[0] = map.init_state
    nodes[-1] = map.goal_state

    map.connect_edge(n, nodes)
    d = Dijkstra(map)
    d.dijkstra(0, -1)
    a = d.get_path(n - 1)
    map.path = a

    edge_index = graph.matrix2list(map.graph)

    print(edge_index)

    print('path is', a)
    # print(np.where(graph.get_label()[:, 1] == 1)[0].shape)

    # print(graph.get_data())
