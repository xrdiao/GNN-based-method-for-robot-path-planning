import torch
from module_3d import MyGnn, NodeEncoder, MapEncoder
from map_3d import maps_3d
import numpy as np
from MyGNN import Guidance
from MyMotionPlan import MyMotionPlan
import pybullet
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from Dijkstra import Dijkstra
from my_graph import my_graph
import BIT
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 记得改


def model_simu(ind, n=400):
    n = 800
    nodes, _ = m.sample_points(n)
    m.nodes = nodes
    m.map = torch.tensor(m.build_map(15).reshape(1, 15, 15, 15), dtype=torch.float)
    m.connect_edge2(m.nodes, int(n/4))

    data = torch.cat(
        (torch.tensor(np.array(nodes), dtype=torch.float),
         torch.tensor(m.goal_state, dtype=torch.float).unsqueeze(0).repeat(len(nodes), 1)), dim=1)
    edge_index = my_gra.matrix2list(m.graph)

    path = p.plan(len(nodes), g, data, edge_index, m.map)

    if path[-1] == m.goal_index:
        m.my_simulation(path, False)
        print('success')
    else:
        print('fail to find path')


def BIT_simu(ind, iter_max=400):
    # 规划
    bit.planning()
    path = bit.ExtractPath()

    if path[0].any() == m.goal_state.any() and len(path) > 1 and path is not None:
        path.reverse()
        m.my_simulation(path)
        print('success')
    else:
        print('fail to find path')


if __name__ == '__main__':
    m = maps_3d(GUI=True,
                model_file='kuka_iiwa/model_0.urdf',
                map_file='maze_files/kukas_7_3000.pkl')
    bit = BIT.BITStar(2, 800, m)

    g = Guidance(node_channel=14, map_chanel=1, obs_channel=6, out_channel=32, batch=1, init_net=False)
    g.to(device)
    g.load_state_dict(torch.load('model/model.pth'))
    g.eval()

    p = MyMotionPlan(m)
    my_gra = my_graph(m)

    while True:
        # ind = np.random.randint(0, 300)
        # print(ind)
        ind = 288 #288,1960
        m.init_problem(ind)
        # print('--------------------- model -------------------------------')
        # model_simu(ind)
        # m.init_problem(ind)
        print('--------------------- BIT -------------------------------')
        BIT_simu(ind)
