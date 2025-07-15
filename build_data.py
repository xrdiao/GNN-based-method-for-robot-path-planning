import matplotlib.pyplot as plt
from torch.utils.data.sampler import WeightedRandomSampler
from my_graph import my_graph
from Dijkstra import Dijkstra
from map_3d import maps_3d
import time
import numpy as np
from tqdm import tqdm
import torch


def rebuild_dataset(size):
    '''
    用来把数据集整合到一个数组中并保存
    '''
    nodes_set = []
    graph_set = []
    map_set = []
    for i in tqdm(range(size)):
        count = i
        node = np.load('data/nodes_set_%d.npy' % count)
        graph = np.load('data/graph_set_%d.npy' % count)
        map_ = np.load('data/map_set_%d.npy' % count)

        if i == 0:
            nodes_set = node
            graph_set = graph
            map_set = map_
        else:
            nodes_set = np.vstack([nodes_set, node])
            graph_set = np.vstack([graph_set, graph])
            map_set = np.vstack([map_set, map_])

    np.save('data/nodes_set', nodes_set)
    np.save('data/graph_set', graph_set)
    np.save('data/map_set', map_set)


def get_dataset():
    # 甚至可以不用这一步，直接在训练的时候生成也可以
    maps = maps_3d(map_file='maze_files/train_3d.pkl')
    dataset = []

    # train_set = torch.load("data/train_set.pth")
    node_set = np.load('data/nodes_set.npy')
    graph_set = np.load('data/graph_set.npy')
    size = len(graph_set)
    g = my_graph(maps)

    for i in tqdm(range(size)):
        # 数据集中，只需要随机图即可
        maps.init_problem(i)
        maps.nodes = node_set[i]
        maps.graph = graph_set[i]
        g.nodes = node_set[i]

        # 获取数据集
        data = g.get_data()
        dataset.append(data)

    train_set = dataset
    train_loader = g.get_loader(train_data=train_set)
    torch.save(train_loader, 'data/train.pth')


if __name__ == '__main__':
    # rebuild_dataset(90)
    get_dataset()

    m = maps_3d(map_file='maze_files/train_3d.pkl')
    map_set = np.load("data/map_set.npy")
    m.map2loader(map_set)

    train = torch.load("data/map.pth")
    print(len(train))
