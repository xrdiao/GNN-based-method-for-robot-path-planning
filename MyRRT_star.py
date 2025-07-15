from MyRRT import MyLeaf
import numpy as np
from map_3d import maps_3d

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


class MyRRT_star:
    def __init__(self, maps):
        self.env = maps
        self.tree = set()
        self.goal = MyLeaf(None, None, self.env.goal_state)
        self.start = MyLeaf(None, None, self.env.init_state)
        self.upper = 4
        self.delta = 3.5
        self.r = self.upper * 1.5
        self.collision_times = 0

    def init_tree(self):
        self.tree = set()
        self.goal = MyLeaf(None, None, self.env.goal_state)
        self.start = MyLeaf(None, None, self.env.init_state)
        self.tree.add(self.start)
        self.collision_times = 0

    def grow_tree(self):
        while True:
            x_rand = self.env.uniform_random()
            if self.env.check_point(x_rand):
                break
        # x_rand = np.random.uniform(low=-1, high=1, size=2)

        father = self.start
        X_near = []  # 存的是邻近节点

        min_dist = self.env.dist(x_rand, self.start.position)
        for node in self.tree:
            dist = self.env.dist(x_rand, node.position)

            if dist <= min_dist:
                min_dist = dist
                father = node

        x_near = father.position
        collision_free = False
        position = []

        if self.env.dist(x_near, x_rand) < self.upper and self.env.check_edge(x_near, x_rand):
            position = x_rand
            self.collision_times = self.collision_times + 1

        else:
            # upper = self.upper
            length = 10
            for i in range(length):
                upper = self.upper * (length - i) / length

                vector = x_rand - x_near
                vector_len = self.env.dist(x_rand, x_near)
                norm_vec = vector / vector_len

                position = x_near + upper * norm_vec
                if self.env.check_edge(x_near, position):
                    self.collision_times = self.collision_times + 1
                    collision_free = True
                    break
            if not collision_free:
                return

        # 选父节点
        min_dist = 100000
        for node in self.tree:
            if self.env.dist(position, node.position) < self.r:
                self.collision_times = self.collision_times + 1
                if self.env.check_edge(position, node.position):
                    dist = self.env.dist(position, node.position) + node.dist
                    X_near.append(node)
                    if dist < min_dist:
                        father = node
                        min_dist = dist

        x_new = MyLeaf(0, father, position, min_dist)
        self.tree.add(x_new)

        # 重新布线
        for k in X_near:
            self.collision_times = self.collision_times + 1
            if k.dist > self.env.dist(position, k.position) + min_dist and self.env.check_edge(position, k.position):
                k.father = x_new

    def near_goal(self, state):
        self.collision_times = self.collision_times + 1
        if self.env.dist(state, self.goal.position) < self.delta and self.env.check_edge(state, self.goal.position):
            return True
        return False

    def find_goal(self, iter_num):
        self.init_tree()
        for _ in range(iter_num):
            self.grow_tree()
            for node in self.tree:
                if self.near_goal(node.position):
                    # print('Find goal!')
                    self.goal.father = node
                    self.goal.root = 0
                    self.tree.add(self.goal)
                    return

    def plot_tree(self, radius=1):
        state = np.array(self.goal.position)
        circle = patches.Circle((state + 1.0), radius=0.01 * radius, facecolor='r', edgecolor='r')
        plt.gca().add_patch(circle)

        plt.axis([0.0, 2.0, 0.0, 2.0])
        for i in self.tree:
            state = np.array(i.position)
            circle = patches.Circle((state + 1.0), radius=0.01 * radius)
            plt.gca().add_patch(circle)

            next_state = np.array(i.father.position)
            path = patches.ConnectionPatch(state + 1.0, next_state + 1.0, 'data', arrowstyle='-',
                                           color='red')
            plt.gca().add_patch(path)
            plt.pause(0.1)
            plt.ioff()

    def get_path(self):
        if self.goal.father is None:
            return [self.goal.position]

        paths = []
        node = self.goal

        while True:
            paths.append(node.position)
            if node.father is None:
                return paths

            node = node.father


if __name__ == '__main__':
    m = maps_3d()
    m.init_problem(200)

    rrt_star = MyRRT_star(m)
    rrt_star.find_goal(300)
    print(len(rrt_star.tree))
    print(rrt_star.collision_times)
    print(rrt_star.get_path())
    print(m.init_state)
    print(m.goal_state)

    print('-----------------------------')
    # m.init_problem(0)
    a = time.time()
    rrt_star.find_goal(1000)
    path = rrt_star.get_path()
    b = time.time()

    print(b - a)
    print(len(rrt_star.tree))
    print(rrt_star.collision_times)
    # rrt_star.plot_tree()
    # plt.show()
