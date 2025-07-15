import torch
import pybullet as p
import time
import numpy as np
from map_3d import maps_3d
from my_graph import my_graph
from MyMotionPlan import MyMotionPlan
from MyGNN import Guidance
import BIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 记得改

m = maps_3d(GUI=True, model_file='ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf')
# m.init_problem(0)
my_gra = my_graph(m)
planer = MyMotionPlan(m)

g = Guidance(node_channel=14, map_chanel=1, obs_channel=6, out_channel=32, batch=1, init_net=False)
g.load_state_dict(torch.load('model/model.pth'))
g.to(device)
g.eval()

# 求逆运动学解
init_pos = [0.28, 0.3, 0.5]
a = p.calculateInverseKinematics(m.model, 5, init_pos)
for i in range(p.getNumJoints(m.model) - 1):
    p.resetJointState(m.model, i, a[i])
a = np.array(a)
a = np.append(a, 0)
m.init_state = a

final_pos = [-0.28, 0.24, 0.2]
a = p.calculateInverseKinematics(m.model, 5, final_pos)
for i in range(len(a)):
    p.resetJointState(m.model, i, a[i])
a = np.array(a)
a = np.append(a, 0)
m.goal_state = a

# 搭建障碍物
size1 = np.array([0.025, 0.085, 0.35])
size2 = np.array([0.8, 0.35, 0.03])

pos = np.array([0, 0.45, 0.35])
pos1 = np.array([-0.43, 0.24, 0.75])
m.create_obs(size1, pos)
m.create_obs(size1, pos1)

pos2 = np.array([-0.8, 0.4, 0.03])

m.create_obs(size2, pos2)

m.obstacles = []
m.obstacles.append((size1, pos))
m.obstacles.append((size1, pos1))
m.obstacles.append((size2, pos2))

size = 15
m.build_map(size)
b = m.map
c = np.where(m.map != 0)
m.map = torch.tensor(m.map, dtype=torch.float).unsqueeze(dim=0)

nodes, _ = m.sample_points(800)
m.nodes = nodes
m.connect_edge2(m.nodes, 400 / 4)

data = torch.cat(
    (torch.tensor(np.array(nodes), dtype=torch.float),
     torch.tensor(m.goal_state, dtype=torch.float).unsqueeze(0).repeat(len(nodes), 1)), dim=1)
edge_index = my_gra.matrix2list(m.graph)
path = planer.plan(len(nodes), g, data, edge_index, m.map)

goal_index = np.where(m.nodes == m.goal_state[0])[0]
init_index = np.where(m.nodes == m.init_state[0])[0]

if path[-1] == goal_index:
    m.my_simulation(path, False, 10)
else:
    print('---------------fail to find ---------------')


# bit = BIT.BITStar(2, 800, m)
# bit.planning()
# path = bit.ExtractPath()
# if path[0].any() == m.goal_state.any() and len(path) > 1 and path is not None:
#     path.reverse()
#     m.my_simulation(path)
#     # print('success')
# else:
#     print('fail to find path')
