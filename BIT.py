import math
import random
import numpy as np
import matplotlib.pyplot as plt
from map_3d import maps_3d
import time


class Node:
    def __init__(self, state):
        self.state = state
        self.parent = None


class Tree:
    def __init__(self, x_start, x_goal):
        self.x_start = x_start
        self.goal = x_goal

        self.r = 4.0
        self.V = set()
        self.E = set()
        self.QE = set()
        self.QV = set()

        self.V_old = set()


class BITStar:
    def __init__(self, eta, iter_max, maps):
        self.env = maps
        self.x_start = Node(self.env.init_state)
        self.x_goal = Node(self.env.goal_state)
        self.eta = eta
        self.iter_max = iter_max

        self.range = self.env.pose_range

        self.Tree = Tree(self.x_start, self.x_goal)
        self.X_sample = set()
        self.g_T = dict()

    def init(self):
        self.x_start = Node(self.env.init_state)
        self.x_goal = Node(self.env.goal_state)

        self.Tree = Tree(self.x_start, self.x_goal)
        self.X_sample = set()
        self.g_T = dict()

        self.Tree.V.add(self.x_start)
        self.X_sample.add(self.x_goal)

        self.g_T[self.x_start] = 0.0
        self.g_T[self.x_goal] = np.inf

        cMin, theta = self.calc_dist_and_angle(self.x_start.state, self.x_goal.state)
        # C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        C = 1
        xCenter = np.array((self.x_start.state + self.x_goal.state) / 2.0)

        return theta, cMin, xCenter, C

    def planning(self):
        theta, cMin, xCenter, C = self.init()

        for k in range(self.iter_max):
            if self.g_T[self.x_goal] != np.inf:
                return 1

            if not self.Tree.QE and not self.Tree.QV:
                if k == 0:
                    m = self.iter_max
                else:
                    m = 200

                self.Prune(self.g_T[self.x_goal])
                self.X_sample.update(self.Sample(m, self.g_T[self.x_goal], cMin, xCenter, C))
                self.Tree.V_old = {v for v in self.Tree.V}
                self.Tree.QV = {v for v in self.Tree.V}
                # self.Tree.r = self.radius(len(self.Tree.V) + len(self.X_sample))

            while self.BestVertexQueueValue() <= self.BestEdgeQueueValue():
                self.ExpandVertex(self.BestInVertexQueue())

            vm, xm = self.BestInEdgeQueue()
            self.Tree.QE.remove((vm, xm))

            if self.g_T[vm] + self.calc_dist(vm, xm) + self.h_estimated(xm) < self.g_T[self.x_goal]:
                actual_cost = self.cost(vm, xm)
                if self.g_estimated(vm) + actual_cost + self.h_estimated(xm) < self.g_T[self.x_goal]:
                    if self.g_T[vm] + actual_cost < self.g_T[xm]:
                        if xm in self.Tree.V:
                            # remove edges
                            edge_delete = set()
                            for v, x in self.Tree.E:
                                if x == xm:
                                    edge_delete.add((v, x))

                            for edge in edge_delete:
                                self.Tree.E.remove(edge)
                        else:
                            self.X_sample.remove(xm)
                            self.Tree.V.add(xm)
                            self.Tree.QV.add(xm)

                        self.g_T[xm] = self.g_T[vm] + actual_cost
                        self.Tree.E.add((vm, xm))
                        xm.parent = vm

                        set_delete = set()
                        for v, x in self.Tree.QE:
                            if x == xm and self.g_T[v] + self.calc_dist(v, xm) >= self.g_T[xm]:
                                set_delete.add((v, x))

                        for edge in set_delete:
                            self.Tree.QE.remove(edge)
            else:
                self.Tree.QE = set()
                self.Tree.QV = set()

    def ExtractPath(self):
        node = self.x_goal
        path = [node.state]

        while node.parent:
            node = node.parent
            path.append(node.state)

        return path

    def Prune(self, cBest):
        self.X_sample = {x for x in self.X_sample if self.f_estimated(x) < cBest}
        self.Tree.V = {v for v in self.Tree.V if self.f_estimated(v) <= cBest}
        self.Tree.E = {(v, w) for v, w in self.Tree.E
                       if self.f_estimated(v) <= cBest and self.f_estimated(w) <= cBest}
        self.X_sample.update({v for v in self.Tree.V if self.g_T[v] == np.inf})
        self.Tree.V = {v for v in self.Tree.V if self.g_T[v] < np.inf}

    def cost(self, start, end):
        if not self.env.check_edge(start.state, end.state):
            return np.inf

        return self.calc_dist(start, end)

    def f_estimated(self, node):
        return self.g_estimated(node) + self.h_estimated(node)

    def g_estimated(self, node):
        return self.calc_dist(self.x_start, node)

    def h_estimated(self, node):
        return self.calc_dist(node, self.x_goal)

    def Sample(self, m, cMax, cMin, xCenter, C):
        # if cMax < np.inf:
        #     return self.SampleEllipsoid(m, cMax, cMin, xCenter, C)
        # else:
        return self.SampleFreeSpace(m)

    # def SampleEllipsoid(self, m, cMax, cMin, xCenter, C):
    #     # 已经找到终点后的子问题采样
    #     r = [cMax / 2.0,
    #          math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
    #          math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
    #     L = np.diag(r)
    #
    #     ind = 0
    #     Sample = set()
    #
    #     while ind < m:
    #         xBall = self.SampleUnitNBall()
    #         x_rand = np.dot(np.dot(C, L), xBall) + xCenter
    #         node = Node(x_rand[(0, 0)], x_rand[(1, 0)])
    #
    #         in_obs = self.env.check_point(node.state)
    #         in_range = self.x_range[0] + delta <= node.x <= self.x_range[1] - delta
    #
    #         if in_obs and in_range:
    #             Sample.add(node)
    #             ind += 1
    #
    #     return Sample

    def SampleFreeSpace(self, m):
        Sample = set()

        ind = 0
        while ind < m:
            node = Node(self.env.uniform_random())
            if not self.env.check_point(node.state):
                continue
            else:
                Sample.add(node)
                ind += 1

        return Sample

    def radius(self, q):
        cBest = self.g_T[self.x_goal]
        lambda_X = len([1 for v in self.Tree.V if self.f_estimated(v) <= cBest])
        radius = 2 * self.eta * (1.5 * lambda_X / math.pi * math.log(q) / q) ** 0.5

        return radius

    def ExpandVertex(self, v):
        self.Tree.QV.remove(v)
        X_near = {x for x in self.X_sample if self.calc_dist(x, v) <= self.Tree.r}

        for x in X_near:
            if self.g_estimated(v) + self.calc_dist(v, x) + self.h_estimated(x) < self.g_T[self.x_goal]:
                self.g_T[x] = np.inf
                self.Tree.QE.add((v, x))

        if v not in self.Tree.V_old:
            V_near = {w for w in self.Tree.V if self.calc_dist(w, v) <= self.Tree.r}

            for w in V_near:
                if (v, w) not in self.Tree.E and \
                        self.g_estimated(v) + self.calc_dist(v, w) + self.h_estimated(w) < self.g_T[self.x_goal] and \
                        self.g_T[v] + self.calc_dist(v, w) < self.g_T[w]:
                    self.Tree.QE.add((v, w))
                    if w not in self.g_T:
                        self.g_T[w] = np.inf

    def BestVertexQueueValue(self):
        if not self.Tree.QV:
            return np.inf

        return min(self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV)

    def BestEdgeQueueValue(self):
        if not self.Tree.QE:
            return np.inf

        return min(self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE)

    def BestInVertexQueue(self):
        if not self.Tree.QV:
            print("QV is Empty!")
            return None

        v_value = {v: self.g_T[v] + self.h_estimated(v) for v in self.Tree.QV}

        return min(v_value, key=v_value.get)

    def BestInEdgeQueue(self):
        if not self.Tree.QE:
            print("QE is Empty!")
            return None

        e_value = {(v, x): self.g_T[v] + self.calc_dist(v, x) + self.h_estimated(x)
                   for v, x in self.Tree.QE}

        return min(e_value, key=e_value.get)

    def calc_dist(self, start, end):
        return self.env.dist(start.state, end.state)

    @staticmethod
    def SampleUnitNBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def calc_dist_and_angle(state1, state2):
        distance = np.sum((state1 - state2) ** 2)
        distance = np.sqrt(distance)

        vector = state2 - state1
        return distance, vector


if __name__ == '__main__':
    m = maps_3d()
    m.init_problem(11)

    a = time.time()
    eta = 2
    iter_max = 200
    bit = BITStar(eta, iter_max, m)
    bit.planning()
    path = bit.ExtractPath()
    length = 0
    if path[0].any() == m.goal_state.any():
        for j in range(len(path) - 1):
            length = length + m.dist(path[j], path[j + 1])
        print(length)
        print(m.goal_state)
        print(path)

    b = time.time()
    print(b-a)
