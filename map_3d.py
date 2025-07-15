import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from multiprocessing.dummy import Pool as ThreadPool
import time
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from Dijkstra import Dijkstra
import math

LIMITE = 1.
inf = 100000


class maps_3d:
    '''
    这个类是用来导入环境的。包括创建问题，创建图，检查是否碰撞等方法。

    目的：存贮地图，起始点，终止点，为规划任务提供环境，包括生成一个概率随机树。
    输入：问题路径，要求包含地图，起始点和终止点
    可输出的项：地图，概率随机树，起始点，终止点
    功能：
        1.概率随机树：随机采样，碰撞检测，k近邻构图
        2.返回torch_geometric.data类的数据类型，以供GNN网络训练

    使用pybullet
    '''

    def __init__(self, model_file='kuka_iiwa/model_0.urdf', map_file='maze_files/kukas_7_3000.pkl', GUI=False):
        self.problem = None
        self.goal_state = None
        self.init_state = None
        self.init_index = None
        self.goal_index = None
        self.map = None
        self.nodes = None
        self.graph = None  # adjacent matrix
        self.path = None
        self.k = 30

        self.obstacles = []
        self.node_set = []
        self.graph_set = []
        self.map_set = []
        self.rank = []
        self.obstacle_times = 0
        self.epoch_i = 0

        self.dim = 3
        self.model_file = model_file

        if GUI:
            p.connect(p.GUI,
                      options='--background_color_red=0.9 --background_color_green=0.9 --background_color_blue=0.9')
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-65, cameraTargetPosition=[0, 0, 0])

        # 装载模型，初始化环境
        self.problems = np.load(map_file, encoding='bytes', allow_pickle=True)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.model = p.loadURDF(model_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.loadURDF("plane100.urdf")
        p.performCollisionDetection()

        # 机械臂的自由度
        self.config_dim = p.getNumJoints(self.model)

        # 获取机械臂电机的转动范围
        self.pose_range = [(p.getJointInfo(self.model, jointId)[8], p.getJointInfo(self.model, jointId)[9]) for
                           jointId in range(p.getNumJoints(self.model))]
        self.bound = np.array(self.pose_range).T.reshape(-1)

        p.setGravity(0, 0, -10)
        self.size = len(self.problems)

    def reset(self):
        '''
        重制数据集，防止栈爆炸
        '''
        self.problem = None
        self.goal_state = None
        self.init_state = None
        self.init_index = None
        self.goal_index = None
        self.map = None
        self.nodes = None
        self.graph = None  # adjacent matrix
        self.path = None

        self.obstacles = []
        self.node_set = []
        self.graph_set = []
        self.map_set = []

    def init_problem(self, index=None):
        '''
        获得起始点，终止点，障碍物，图的信息需要额外输入
        Args:
            index: 第i个问题信息

        '''
        if index is None:
            index = self.epoch_i

        self.obstacles, self.init_state, self.goal_state, self.path = self.problems[index]

        self.epoch_i += 1
        self.epoch_i = self.epoch_i % self.size

        p.resetSimulation()
        self.model = p.loadURDF(self.model_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        p.loadURDF("plane100.urdf")
        p.performCollisionDetection()

        for size, position in self.obstacles:
            self.create_obs(size, position)

        return self.get_problem()

    def get_problem(self, size=15):
        self.problem = {
            'maze': np.array(self.build_map(size)).astype(float),
            'init_state': self.init_state,
            'goal_state': self.goal_state
        }
        return self.problem

    def load_nodes(self, nodes):
        self.nodes = nodes
        self.init_state = nodes[0]
        self.goal_state = nodes[1]

    def uniform_random(self):
        sample = np.random.uniform(low=np.array(self.pose_range)[:, 0], high=np.array(self.pose_range)[:, 1],
                                   size=self.config_dim)
        return sample

    def random_init_goal_state(self, n, change=False):
        '''
        从nodes中随机选取两个点当起始点和终止点
        '''
        init_index = np.random.randint(0, n, 1)
        goal_index = np.random.randint(0, n, 1)

        while goal_index == init_index:
            goal_index = np.random.randint(0, n, 1)

        if change:
            self.init_state = self.nodes[init_index]
            self.goal_state = self.nodes[goal_index]

        return init_index[0], goal_index[0]

    def sample_points(self, n):
        '''
            随机采样n个点，并返回数组
            samples为自由空间中的点，neg_samples是采在障碍物中的点，可以作为负样本输出
        '''
        samples = []
        neg_samples = []
        for i in range(n):
            j = 0
            while True:
                sample = self.uniform_random()
                if self.check_point(sample):
                    samples.append(sample)
                    break
                else:
                    neg_samples.append(sample)
                if j >= 15000:
                    return None, None
                j += 1

        self.init_index, self.goal_index = self.random_init_goal_state(len(samples))

        samples[self.init_index] = self.init_state
        samples[self.goal_index] = self.goal_state

        return samples, neg_samples

    def dist(self, state1, state2):
        '''
        用来计算两点间的欧式距离
        '''
        state2 = np.maximum(state2, np.array(self.pose_range)[:, 0])
        state2 = np.minimum(state2, np.array(self.pose_range)[:, 1])

        distance = np.sum((state1 - state2) ** 2)
        distance = np.sqrt(distance)

        return distance

    # def heuristic_dist(self, state1, state2, alpha=0.5):
    #     '''
    #     用来计算两点的估计距离，即d = (1-a) * dist(state1, state2) + a * goal_state-state1
    #     Args:
    #         alpha: 权重
    #     '''
    #     return (1 - alpha) * self.dist(state1, state2) + alpha * self.dist(self.goal_state, state1)

    def _dist(self, state1, state2):
        return np.sum(np.abs(state1 - state2))

    def _connect(self, i):
        dists = [self._dist(self.nodes[i], self.nodes[j]) for j in range(len(self.nodes))]
        lim = np.sum(dists, axis=0) / len(dists)
        a = np.where(dists < lim)[0]
        if len(a) > self.k:
            self.graph[i, a[:self.k]] = 1
            self.graph[a[:self.k], i] = 1
        else:
            self.graph[i, a] = 1
            self.graph[a, i] = 1

    def connect_edge2(self, nodes, k=None):
        '''
        返回一张k邻域图，不进行碰撞检测
        '''

        self.k = int(k)
        n = len(nodes)
        self.graph = np.ones([n, n]) * inf
        # self.graph[self.goal_index][self.init_index] = 1
        # self.graph[self.init_index][self.goal_index] = 1
        self.nodes = nodes
        self.node_set.append(self.nodes)

        pool = ThreadPool(5)
        pool.map(self._connect, np.arange(n), self.k)
        pool.close()
        pool.join()

        self.graph_set.append(self.graph)  # 图集

    def quick_sort(self, array, start, end):
        '''
            用快速排序来给节点间的启发式距离进行排序。
            快速排序是基于二分法的排序法。
            快速排序的运行逻辑为：先从数组中选取一个’中间值‘，然后通过两个’指针‘分别指向起始和终点，通过两个指针的互相靠近进行遍历，在指针靠近
            的时候，将比中间值小的放在它的左边，大的放在右边，循环调用，最后实现排序(从小到大)
        '''
        arr = array.copy()
        if start >= end:
            return
        mid_data, rank_s, left, right = arr[start], self.rank[start], start, end
        while left < right:
            # 比较右边的数值和中间值，并进行交换
            while arr[right] >= mid_data and left < right:
                right -= 1
            arr[left] = arr[right]
            self.rank[left] = self.rank[right]
            # 比较左边的数值和中间值，并进行交换
            while arr[left] < mid_data and left < right:
                left += 1
            arr[right] = arr[left]
            self.rank[right] = self.rank[left]
        # 把中间值重新嵌入进数组
        arr[left] = mid_data
        self.rank[left] = rank_s
        # 循环调用
        self.quick_sort(arr, start, left - 1)
        self.quick_sort(arr, left + 1, end)

    def connect_edge(self, nodes, k=5):
        '''

        Args:
            n: size of graph
            nodes: vertices of graph

        Returns:
            adjacent Matrix

        this def is used to connect nodes without collision

        '''
        n = len(nodes)
        self.graph = np.ones([n, n]) * inf
        self.nodes = nodes
        self.node_set.append(self.nodes)
        k_connect = np.zeros(n)

        for i in range(n):
            dists = [self.dist(self.nodes[i], self.nodes[j]) for j in range(n)]
            self.rank = np.arange(len(dists))
            self.quick_sort(dists, 0, len(dists) - 1)

            for j in self.rank:
                if i != j and self.graph[i][j] == inf and k_connect[i] < k and k_connect[j] < k:
                    self.graph[i][j] = dists[j]
                    self.graph[j][i] = dists[j]
                    k_connect[i] += 1
                    k_connect[j] += 1

                    if k_connect[i] >= k:
                        break

        self.graph_set.append(self.graph)  # 图集

    def process_nodes(self, label, raw_nodes):
        # 从model获得的优先级来选择节点
        first = np.array(np.where(label[:, 0] == 1))[0]
        second = np.array(np.where(label[:, 1] == 1))[0]
        nodes_index = np.hstack([first, second]).astype(int)
        new_nodes = [raw_nodes[i] for i in nodes_index]
        # raw_nodes[nodes_index]
        return new_nodes

    def load_graph(self, graph):
        self.graph = graph

    def goal_area(self, state):
        return self.dist(state, self.goal_state) <= 0.01

    def check_point(self, state):
        '''
        Returns:
            True - 没有碰撞
            False - 发生碰撞
        '''
        # 这个函数是用来检查输入的state是否会发生碰撞，所以使用resetJointState来使模型直接到达目标点，而不是通过仿真让其‘缓慢到达’。
        for i in range(p.getNumJoints(self.model)):
            p.resetJointState(self.model, i, state[i])

        p.performCollisionDetection()
        p.stepSimulation()

        if p.getContactPoints(self.model):  # p.getContact 碰撞返回True, 否则返回False
            return False
        else:
            return True  # 还需要判断state的合理性

    def find_path(self, end):
        '''
        用来查找从init_state到goal_state的路径，需要让maps_3d类获得nodes和graph信息
        Returns:
            路径
        '''
        d = Dijkstra(self)
        d.dijkstra(0, end)
        return d.get_path(end)

    def check_segment(self, start, end):
        '''
            用迭代的方法来检查边是否会产生碰撞
        Returns:
            True: 没有发生碰撞
            False： 发生碰撞
        '''
        if np.sum(np.abs(start - end)) > 0.1:
            mid = (start + end) / 2.0
            if not self.check_point(mid):
                return False
            return self.check_segment(start, mid) and self.check_segment(mid, end)
        return True

    def check_edge(self, state, new_state):
        '''
        Returns:
            True: 没有发生碰撞
            False： 发生碰撞
        '''
        # self.k=0
        self.obstacle_times += 1
        if not self.check_point(state) or not self.check_point(new_state):  # not check表示为：发生碰撞则返回False
            return False
        return self.check_segment(state, new_state)

    def create_obs(self, size, position):
        '''
        用来在仿真环境中生成障碍物
        Args:
            size: 障碍物的长宽高/2，看函数中这个参数的命名为 half extents，所以是一半
            position: 障碍物中心点的坐标（也可以说是基点的坐标）

        Returns:
            障碍物id
        '''
        obs_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        obs_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                           rgbaColor=[0, 0, 0, 0.7],
                                           halfExtents=size)
        obs_id = p.createMultiBody(baseMass=0,
                                   baseCollisionShapeIndex=obs_collision_id,
                                   baseVisualShapeIndex=obs_shape_id,
                                   basePosition=position)
        return obs_id

    def build_map(self, size=15):
        '''
        过程如下：
        1.设置坐标网格
        2.将障碍物映射到网格中
        3.返回地图
        障碍物x:[-1,1]和地图y:[0,size-1]的映射如下：
        x = y*2/(size-1)-1

        Args:
            size: int类型，表示地图的大小[size, size, size]

        Returns:
            一个表示地图状态的矩阵，

        '''
        base = np.linspace(-1., 1., num=size)
        point = np.meshgrid(base, base, base)
        point = np.concatenate((point[0].reshape(-1, 1), point[1].reshape(-1, 1), point[2].reshape(-1, 1)), axis=-1)
        obs_map = np.zeros(point.shape[0]).astype(bool)

        for obs in self.obstacles:
            # 获取对角线坐标值
            obs_size, obs_pos = obs
            dig_low, dig_high = obs_pos - obs_size, obs_pos + obs_size
            dig_low[2], dig_high[2] = dig_low[2] - 0.4, dig_high[2] - 0.4  # 调整障碍物在构建的地图中的高度

            # 获取障碍物的坐标矩阵
            obs_coordinate = []
            for i in range(3):
                obs_mask = np.zeros(size)
                # 将障碍物的xyz在坐标中表示出来
                obs_mask[
                max(int((dig_low[i] + 1) * (size - 1) / 2.), 0):min(1 + int((dig_high[i] + 1) * (size - 1) / 2.),
                                                                    1 + size - 1)] = 1
                obs_coordinate.append(obs_mask.astype(bool))

            # 获取障碍物在地图中的真实位置
            current_obs = np.meshgrid(*obs_coordinate)
            current_obs = np.concatenate(
                (current_obs[0].reshape(-1, 1), current_obs[1].reshape(-1, 1), current_obs[2].reshape(-1, 1)), axis=-1)
            # 更新地图
            current_map = np.all(current_obs, axis=-1)
            obs_map = obs_map | current_map

        obs_map = obs_map.reshape((size, size, size))
        self.map = obs_map
        self.map_set.append(self.map)  # 地图集
        return obs_map

    def plot_cube(self, size=15, fig=None):
        if fig is None:
            fig = plt.figure()
        ax = fig.gca(projection='3d')

        for obs in self.obstacles:
            # 获取对角线坐标值
            x, y, z = np.indices((size, size, size))

            obs_size, obs_pos = obs
            dig_low, dig_high = obs_pos - obs_size, obs_pos + obs_size
            dig_low[2], dig_high[2] = dig_low[2] - 1.2, dig_high[2] - 1.2

            cube = (x >= max(int((dig_low[0] + 1) * (size - 1) / 2), 0)) & \
                   (y >= max(int((dig_low[1] + 1) * (size - 1) / 2), 0)) & \
                   (z >= max(int((dig_low[2] + 1) * (size - 1) / 2), 0)) & \
                   (x <= min(int((dig_high[0] + 1) * (size - 1) / 2), size - 1)) & \
                   (y <= min(int((dig_high[1] + 1) * (size - 1) / 2), size - 1)) & \
                   (z <= min(int((dig_high[2] + 1) * (size - 1) / 2), size - 1))
            ax.voxels(cube, shade=False)
        return fig

    def plot_node(self, states, size=15, fig=None, label=None, color='y'):
        if fig is None:
            fig = plt.figure()
        ax = fig.gca(projection='3d')

        if states is None:
            return fig

        if label is None:
            for index in range(len(states)):
                for i in range(p.getNumJoints(self.model)):
                    p.resetJointState(self.model, i, states[index][i])
                node_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])
                node_pos = (node_pos + 1) * (size - 1) / 2
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], c=color)
        else:
            first = np.array(np.where(label[:, 0] == 1))[0]
            second = np.array(np.where(label[:, 1] == 1))[0]
            third = np.array(np.where(label[:, 2] == 1))[0]

            for j in first:
                for i in range(p.getNumJoints(self.model)):
                    p.resetJointState(self.model, i, states[j][i])
                node_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])
                node_pos = (node_pos + 1) * (size - 1) / 2
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], c='r')

            for j in second:
                for i in range(p.getNumJoints(self.model)):
                    p.resetJointState(self.model, i, states[j][i])
                node_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])
                node_pos = (node_pos + 1) * (size - 1) / 2
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], c='g')

            for j in third:
                for i in range(p.getNumJoints(self.model)):
                    p.resetJointState(self.model, i, states[j][i])
                node_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])
                node_pos = (node_pos + 1) * (size - 1) / 2
                ax.scatter(node_pos[0], node_pos[1], node_pos[2], c='b')

        return fig

    def plot_map(self, states=None, size=15, label=None, color='y'):
        '''
            用来绘制np的数组图，需要确保states的输入或者maps_3d的nodes两者至少有一个存在
        '''
        if states is None:
            states = self.nodes
        fig = self.plot_cube()
        self.plot_node(states, fig=fig, label=label, color=color)
        plt.savefig('fig/result')
        plt.show()

    def two2N(self, state1, state2, N):
        step = (state2 - state1) / (N - 1)
        res_list = [state1]
        for i in range(N - 2):
            res_list.append(res_list[i] + step)
        res_list.append(state2)
        return res_list

    def my_simulation(self, path, if_list=True, duration=5):
        '''
        用来pybullet仿真演示，需要确保给maps_3d类传入nodes，path，并且使用了init_problem函数
        '''
        a = []
        N = 20
        text_size = 1.5
        [p.resetJointState(self.model, i, self.init_state[i]) for i in range(p.getNumJoints(self.model))]
        p.addUserDebugText(text='Init state',
                           textPosition=list(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0]),
                           textColorRGB=[1, 0, 0], textSize=text_size)
        obs_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                           rgbaColor=[1, 0, 0, 0.7],
                                           radius=0.05)
        p.createMultiBody(baseMass=0,
                          baseVisualShapeIndex=obs_shape_id,
                          basePosition=list(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0]))

        time.sleep(1)

        for i in range(len(path) - 1):
            if not if_list:
                res_list = self.two2N(self.nodes[path[i]], self.nodes[path[i + 1]], N)
            else:
                res_list = self.two2N(path[i], path[i + 1], N)
            a = a + res_list

        for i in tqdm(range(len(a))):
            recent_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])

            for _ in range(50):
                p.setJointMotorControlArray(self.model, list(range(self.config_dim)), p.POSITION_CONTROL,
                                            a[i])
                time.sleep(1 / 480)
                p.stepSimulation()

            next_pos = np.array(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0])
            p.addUserDebugLine(recent_pos, next_pos, [1, 0, 0], 3)
            if i % 5 == 0:
                obs_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                   rgbaColor=[1, 0, 0, 0.7],
                                                   radius=0.05)
                p.createMultiBody(baseMass=0,
                                  baseVisualShapeIndex=obs_shape_id,
                                  basePosition=next_pos)

        p.addUserDebugText(text='Goal state',
                           textPosition=list(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0]),
                           textColorRGB=[1, 0, 0], textSize=text_size)
        obs_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                           rgbaColor=[1, 0, 0, 0.7],
                                           radius=0.05)
        p.createMultiBody(baseMass=0,
                          baseVisualShapeIndex=obs_shape_id,
                          basePosition=list(p.getLinkState(self.model, p.getNumJoints(self.model) - 1)[0]))
        time.sleep(duration)

    def save_data(self):
        np.save('data/nodes_set', self.node_set)
        np.save('data/graph_set', self.graph_set)
        np.save('data/map_set', self.map_set)

    def map2loader(self, map_set):
        datalist = []
        for i in range(len(map_set)):
            x = torch.tensor(map_set[i].reshape(1, 15, 15, 15), dtype=torch.float)
            data = Data(x)
            datalist.append(data)
        map_loader = DataLoader(datalist, batch_size=1, shuffle=True, drop_last=True)
        torch.save(map_loader, 'data/map.pth')
        return map_loader

    def check_sim(self, state):
        j = 0
        while True:
            for i in range(p.getNumJoints(self.model)):
                p.resetJointState(self.model, i, state[i])
            if j == 50:
                break
            time.sleep(1 / 60)
            p.stepSimulation()
            j += 1


if __name__ == '__main__':
    m = maps_3d(GUI=False, map_file='maze_files/kukas_7_3000.pkl')
    sizes = 200
    count = 0

    # 构建数据集
    for l in tqdm(range(300)):
        m.init_problem(l)
        node, _ = m.sample_points(sizes)
        # print(m.init_index, m.init_state)
        if node is None:
            continue
        else:
            m.connect_edge2(node, 30)

    # m.init_problem(0)
    # nodes, _ = m.sample_points(n)
    # m.connect_edge2(nodes, 30)

    # m.plot_map(nodes)

    # while True:
    #     p.stepSimulation()
    m.save_data()
    print('sucess')
