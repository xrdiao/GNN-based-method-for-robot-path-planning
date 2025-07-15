from map_3d import maps_3d
import numpy as np
from MyGNN import Guidance
import torch
from Dijkstra import Dijkstra


class MyMotionPlan:
    def __init__(self, maps_3d):
        self.m = maps_3d
        self.nodes = None
        self.map = None
        self.rank = None
        self.dist = None

        self.init_state = None
        self.goal_state = None
        self.g = []  # 已经搜索过的目标
        self.flag = None
        self.edge = []

        self.obstacle_times = 0
        self.guidance = None

    def reset_para(self):
        self.nodes = np.array(self.m.nodes)
        self.map = self.m.map
        self.rank = np.arange(len(self.nodes))
        self.dist = None

        self.init_state = self.m.init_state
        self.goal_state = self.m.goal_state
        self.g = []  # 已经搜索过的目标
        self.flag = np.zeros(len(self.nodes))
        self.edge = []

        self.guidance = None

    def quick_sort(self, arr, start, end):
        '''
            用快速排序来给节点间的启发式距离进行排序。
            快速排序是基于二分法的排序法。
            快速排序的运行逻辑为：先从数组中选取一个’中间值‘，然后通过两个’指针‘分别指向起始和终点，通过两个指针的互相靠近进行遍历，在指针靠近
            的时候，将比中间值小的放在它的左边，大的放在右边，循环调用，最后实现排序(从小到大)
        '''
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

    def get_dist(self, index):
        '''
        用来获得其他节点到某节点的距离并进行排序
        Args:
            index: 某节点的序号

        Returns:
            更新class中的变量dist和rank

        '''
        # 自己不能到自己
        dist = self.guidance[index].cpu().detach().numpy()
        dist[index] = 0
        self.quick_sort(dist, 0, len(self.nodes) - 1)

    def get_next(self, current_index):
        '''
        -----------------------------------------------------------------------------------------------
        GNN学习的是这个过程！学的就是如何获得next_state

        根据距离当前节点的启发式距离的大小，从大到小进行碰撞检测，如果检测成功，则返回下一个节点
        Args:
            current_index: 当前节点的序号

        Returns:
            next_state：下一个节点的状态
            None：失败

        '''
        self.rank = np.arange(len(self.nodes))
        self.get_dist(current_index)
        for i in range(int(self.m.k / 1)):
            next_index = self.rank[len(self.nodes) - 1 - i]

            if self.guidance[current_index][next_index].item() != 0:
                self.guidance[current_index][next_index] = 0
                next_state = self.nodes[next_index]

                self.obstacle_times += 1
                if self.flag[self.rank[len(self.nodes) - 1 - i]] == 0 and self.m.check_edge(self.nodes[current_index],
                                                                                            next_state):
                    return next_state, next_index
            else:
                break
        return None, None

    def plan(self, loop, guidance, nodes, edge_index, env, k=30):
        '''
        kono 我的规划器哒，具体的逻辑在那张写有程序伪代码的纸上，简单来说就是一个简单的贪婪搜索算法
        Args:
            loop：循环次数
            guidance：Guidance类
            k: k-NN的k
        Returns:
            edge：路径
            None：搜索失败
        '''
        self.reset_para()

        # current_state = self.init_state
        state_index = self.m.init_index
        self.edge.append(state_index)
        goal_index = self.m.goal_index
        self.flag[state_index] = 1
        i = 0

        # 遍历完所有节点仍没有可行路径，返回None
        for _ in range(loop):
            path = self.nodes[self.edge]
            # 这个算法本质就是搜索算法，通过修改guidance的生成方式就可以所有的同类型的规划算法。（大概）
            self.guidance = guidance(nodes, edge_index, env,
                                     torch.tensor(np.array(path), dtype=torch.float).detach())

            # 这个state_index指的就是上一个节点的序号
            next_state, state_index = self.get_next(state_index)

            # 表明当前节点已经没路走了，返回None（可能陷入局部最小值）
            if next_state is None:
                # print('next state is none')
                return self.edge

            # 对数据进行更迭
            self.edge.append(state_index)
            self.flag[state_index] = 1

            if goal_index in self.edge:
                return self.edge

            i += 1
            if i == loop:
                return self.edge


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 记得改


def My_train(g, m, p, d, opt, epoch):
    i = 0
    losses = 0
    success = 0

    for data, env in zip(train, maps):
        print('----------------i =', i, '------------------')
        m.init_problem(i)

        m.map = map_set[i]
        m.nodes = node_set[i]
        m.graph = graph_set[i]
        g.env_coder = None

        print(np.where(m.nodes == m.init_state[0]))

        m.init_index = np.where(m.nodes == m.init_state[0])[0][0]
        m.goal_index = np.where(m.nodes == m.goal_state[0])[0][0]

        print('init_index:', m.init_index, 'goal_index', m.goal_index)

        loop = np.random.randint(1, 10, 1)[0]
        print('loop = ', loop)
        path_guidance = p.plan(loop, g, data.x, data.edge_index, env.x)  # 由planner动态生成的路径

        i += 1

        if path_guidance[-1] == m.goal_index:
            success += 1
            print('find path successfully, path:', path_guidance)
            a = [m.check_edge(m.nodes[path_guidance[i]], m.nodes[path_guidance[i + 1]]) for i in
                 range(len(path_guidance) - 1)]
            print(a)
            print('success rate:', success / i)
            continue

        else:
            near_path = np.where(m.graph[path_guidance[-1]] < 20)  # near_path这里有过修改
            guidance = g(data.x, data.edge_index, env.x,
                         torch.tensor(np.array(m.nodes[path_guidance]), dtype=torch.float).detach())

            d.dijkstra(path_guidance[-1], m.goal_index, p.flag)  # path_guidance[-1]
            path_d = d.get_path(m.goal_index)  # 由Dijkstra生成的路径

            if path_d is None:
                print('fail to find path, path_guidance =', path_guidance)
                continue

            else:
                print('path guidance =', path_guidance)
                path_d = path_d[:-1]
                list.reverse(path_d)
                print('path_d =', path_d)
                print('next edge:', [path_guidance[-1], path_d[0]], guidance[path_guidance[-1]][path_d[0]].item(),
                      torch.max(guidance[path_guidance[-1]]).item())

                next_point = np.where(near_path[0] == path_d[0])

                # if near_path[0][next_point] == near_path[0][-1]:
                # guidance的索引为边的索引
                loss = - \
                    guidance[[np.ones(len(near_path[0]), dtype=int) * path_guidance[-1], near_path[0]]].log_softmax(
                        dim=0)[
                        next_point]  # 表示为已经搜索过的最后一个点要去的下一个点的引导值
                loss.to(device)
                loss.backward()

                print('loss :', loss.item(), 'grad :',
                      torch.sum(g.decode[0].weight).item() / len(g.decode[0].weight))
                losses += loss

                opt.step()
                opt.zero_grad()

    torch.save(g.state_dict(), 'model/g{}.pth'.format(epoch))
    return losses, success


if __name__ == '__main__':
    node_set = np.load('data/nodes_set.npy')
    graph_set = np.load('data/graph_set.npy')
    map_set = np.load('data/map_set.npy')
    # samples = np.load('data/samples.npy')
    train = torch.load("data/train.pth")
    maps = torch.load("data/map.pth")

    epoches = 10
    g = Guidance(node_channel=14, map_chanel=1, obs_channel=6, out_channel=32, batch=1, init_net=False)
    g.to(device)
    g.load_state_dict(torch.load('model/model.pth'))

    m = maps_3d(map_file='maze_files/kukas_7_3000.pkl')
    p = MyMotionPlan(m)
    d = Dijkstra(m)
    opt = torch.optim.Adam(g.parameters(), lr=0.001, weight_decay=0.001)
    successes = []

    for epoch in range(epoches):
        losses, success = My_train(g, m, p, d, opt, epoch)
        successes.append(success)
        np.save('data/success_rate', successes)
        print('total loss =', losses)
        print('total success =', success)
