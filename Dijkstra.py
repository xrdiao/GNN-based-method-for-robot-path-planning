import numpy as np
from maps import maps as mp

inf = 100000


class Dijkstra:
    '''
    传入的maps为maps类，而不是array数组
    '''

    def __init__(self, maps):
        self.maps = maps
        self.map = None
        self.size = 0
        self.path = None
        self.flag = None
        self.dist = None
        self.up_v = 0
        self.num_flag = 0
        self.start = -1
        self.nodes = None
        self.obstacle_times = 0

    def reset_para(self):
        self.map = self.maps.graph
        self.nodes = self.maps.nodes
        self.size = len(self.map)
        self.path = np.ones(self.size) * -1
        self.flag = np.zeros(self.size)
        self.dist = np.zeros(self.size)
        self.num_flag = 0
        self.up_v = 0

    def dijkstra(self, start, end, flag=None):
        self.reset_para()
        if flag is not None:
            self.flag = flag

        self.start = start
        self.up_v = start
        self.flag[start] = 1
        self.num_flag += 1

        # init dist
        self.dist = self.map[start]

        # dijkstra
        while True:
            # search the min dist
            min_dist = 0

            for i in range(self.size):
                self.obstacle_times += 1
                if self.dist[i] < self.dist[min_dist] and self.flag[i] == 0 \
                        and self.maps.check_edge(self.nodes[i], self.nodes[min_dist]):
                    min_dist = i
            self.flag[min_dist] = 1
            self.num_flag += 1
            self.up_v = min_dist

            # update dist and path
            for i in range(self.size):
                if self.flag[i] == 0 and self.map[self.up_v][i] + self.dist[self.up_v] < self.dist[i]:
                    self.dist[i] = self.map[self.up_v][i] + self.dist[self.up_v]
                    self.path[i] = self.up_v

            # check breakpoint
            if self.size == self.num_flag or self.path[end] != -1:
                # print('Have searched the entire graph.')
                break

    def get_path(self, end):
        if self.start == -1:
            print('you do not input the start point')
            return None
        if end > self.size - 1:
            print('the end point is not in graph')
            return None

        self.path[self.start] = -1
        path = [end]
        for i in range(self.size):
            path.append(int(self.path[int(path[i])]))
            if self.dist[end] == inf:
                # print('no path')
                return None
            if self.path[path[i]] == -1:
                path[-1] = self.start
                return path
