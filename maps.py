import numpy as np
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch_geometric.loader import DataLoader

LIMITE = 1.
inf = 100000


class maps:
    '''
    这个类是用来导入环境的。包括创建问题，创建图，检查是否碰撞等方法。

    目的：存贮地图，起始点，终止点，为规划任务提供环境，包括生成一个概率随机树。
    输入：问题路径，要求包含地图，起始点和终止点
    可输出的项：地图，概率随机树，起始点，终止点
    功能：
        1.概率随机树：随机采样，碰撞检测，k近邻构图
        2.返回torch_geometric.data类的数据类型，以供GNN网络训练
    '''

    def __init__(self, map_file='maze_files/train.pkl'):
        self.problem = None
        with np.load(map_file) as f:
            self.maps = f['maps']
            self.init_states = f['init_states']
            self.goal_states = f['goal_states']
        self.goal_state = None
        self.init_state = None
        self.map = None
        self.nodes = None
        self.graph = None  # adjacent matrix
        self.path = None

        self.problems = {'maps': self.maps, 'init_states': self.init_states, 'goal_states': self.goal_states}
        self.size = len(self.maps)  # 数据集大小
        self.width = len(self.maps[0])  # 一张地图的大小
        self.indexs = range(self.size)  # 数据集中的地图索引集
        self.obstacles = []
        self.node_set = []
        self.obstacle_times = 0
        self.epoch_i = 0

    def init_problem(self, index=None):
        if index is None:
            index = self.epoch_i

        self.map = self.maps[index]
        self.init_state = self.init_states[index]
        self.goal_state = self.goal_states[index]

        self.epoch_i += 1
        self.epoch_i = self.epoch_i % self.size

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] == 1:
                    self.obstacles.append([i, j])

        return self.get_problem()

    def get_problem(self):
        self.problem = {
            'maze': self.map,
            'init_state': self.init_state,
            'goal_state': self.goal_state
        }
        return self.problem

    def uniform_random(self):
        sample = np.random.uniform(low=-LIMITE, high=LIMITE, size=2)
        return sample

    def sample_points(self, n):
        '''
            随机采样n个点，并返回数组
            samples为自由空间中的点，neg_samples是采在障碍物中的点，可以作为负样本输出
        '''
        samples = []
        neg_samples = []
        for i in range(n):
            while True:
                sample = self.uniform_random()
                if self.check_point(sample):
                    samples.append(sample)
                    break
                else:
                    neg_samples.append(sample)

        samples[0] = self.init_state
        samples[-1] = self.goal_state

        return samples, neg_samples

    def dist(self, state1, state2):
        distance = (state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2
        distance = np.sqrt(distance)
        return distance

    def build_graph(self, n):
        '''

        Args:
            n: size of graph

        Returns:
            None

        but this def has some problems, as the planner can't work well
        '''
        nodes, _ = self.sample_points(n)
        nodes[0] = self.init_state
        nodes[-1] = self.goal_state
        self.connect_edge(n, nodes)

    def connect_edge(self, n, nodes):
        '''

        Args:
            n: size of graph
            nodes: vertices of graph

        Returns:
            adjacent Matrix

        this def is used to connect nodes

        '''
        # 构建邻接矩阵
        self.graph = np.ones([n, n]) * inf
        self.nodes = nodes
        self.node_set.append(self.nodes)
        # 没有自环，没有使用k近邻，只有在附近的点会相连
        for i in range(n):
            for j in range(n):
                dist = self.dist(nodes[i], nodes[j])
                if i != j and dist < LIMITE / 2 and self.check_edge(nodes[i], nodes[j]):
                    self.obstacle_times += 1
                    self.graph[i][j] = dist

    def process_nodes(self, label, raw_nodes):
        # 从model获得的优先级来选择节点
        first = np.array(np.where(label[:, 0] == 1))[0]
        second = np.array(np.where(label[:, 1] == 1))[0]
        nodes_index = np.hstack([first, second])
        return raw_nodes[nodes_index]

    def load_graph(self, graph):
        self.graph = graph

    def get_graph(self):
        if self.graph is None:
            return self.build_graph(100)
        return self.graph

    def goal_area(self, state):
        return self.dist(state, self.goal_state) <= 0.01

    def check_point(self, state, map_width=15):
        '''
            将[-1,1]的state值（double值）转成map中的坐标（int）,然后返回该state是否在障碍物中,1为不在，0为在(LIMITE==1)
        '''
        self.obstacle_times += 1
        coordinate = ((state + 1) * map_width / 2).astype(int)
        return self.map[coordinate[0], coordinate[1]] == 0  # 还需要判断state的合理性

    def check_segment(self, start, end, map_width=15):
        '''
            用迭代的方法来检查边是否会产生碰撞，2维
        '''
        coo_s = (start * self.width).astype(int)
        coo_e = (end * self.width).astype(int)

        if np.sum(np.abs(coo_s - coo_e)) > 1 and np.sum(np.abs(start - end)) > 1e-5:
            mid = (start + end) / 2.0
            if not self.check_point(mid):
                return False
            return self.check_segment(start, mid) and self.check_segment(mid, end)

        return True

    def check_edge(self, state, new_state):
        assert state.shape == new_state.shape
        # self.k=0
        if not self.check_point(state) or not self.check_point(new_state):
            return False
        return self.check_segment(state, new_state)

    def plot_map(self):
        rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0, linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        map = self.map

        map_width = self.width
        d_x = 2.0 / map_width
        d_y = 2.0 / map_width

        for i in range(map_width):
            for j in range(map_width):
                if map[i][j] > 0:
                    rect = patches.Rectangle((d_x * i, d_y * j), d_x, d_y, linewidth=1, edgecolor='#black',
                                             facecolor='#black')
                    plt.gca().add_patch(rect)

        plt.axis([0.0, 2.0, 0.0, 2.0])
        # plt.subplots_adjust(left=-0., right=1.0, top=1.0, bottom=-0.)
        return plt

    def plot_nodes(self, state, title=' ', radius=1, color='#e6550d', label=None):
        if len(state) == 2:
            circle = patches.Circle((state + 1.0), radius=0.01 * radius, edgecolor=color, facecolor=color)
            plt.gca().add_patch(circle)
            plt.annotate(title, state + 1, color='black', backgroundcolor=(1., 1., 1., 0.), fontsize=15)
        else:
            state = np.array(state)
            # label = label.numpy()
            first = np.array(np.where(label[:, 0] == 1))[0]
            second = np.array(np.where(label[:, 1] == 1))[0]
            third = np.array(np.where(label[:, 2] == 1))[0]

            for i in first:
                cir = patches.Circle((state[i] + 1.0), radius=0.01 * radius, edgecolor='r', facecolor='r')
                plt.gca().add_patch(cir)
                plt.annotate(title, state[i] + 1, color='black', backgroundcolor=(1., 1., 1., 0.), fontsize=20)

            for i in second:
                cir = patches.Circle((state[i] + 1.0), radius=0.01 * radius, edgecolor='g', facecolor='g')
                plt.gca().add_patch(cir)

            for i in third:
                cir = patches.Circle((state[i] + 1.0), radius=0.01 * radius, edgecolor='b', facecolor='b')
                plt.gca().add_patch(cir)

    def plot_edge(self, state, next_state):
        path = patches.ConnectionPatch(state + 1.0, next_state + 1.0, 'data', arrowstyle='-',
                                       color='red')
        plt.gca().add_patch(path)

    def map2loader(self, n, fail_index):
        datalist = []
        for i in range(n):
            if i not in fail_index:
                x = torch.tensor(self.maps[i].reshape(1, 15, 15), dtype=torch.float)
                data = Data(x)
                datalist.append(data)
        map_loader = DataLoader(datalist, batch_size=1, shuffle=True, drop_last=True)
        torch.save(map_loader, 'data/map.pth')
        return map_loader

    def save_nodes(self):
        np.save('data/nodes', self.node_set)
