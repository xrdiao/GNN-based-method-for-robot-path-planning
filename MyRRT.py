import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class MyLeaf:
    def __init__(self, root, father, position, dist=0):
        self.root = root
        self.father = father
        self.position = position
        self.dist = dist


class MyRRT:
    def __init__(self):
        self.upper = 0.05
        self.m = np.zeros([15, 15])
        self.rank = []
        self.tree = [MyLeaf(0, 0, [0, 0])]
        self.goal = MyLeaf(-1, -1, [0.5, 0.5])
        self.delta = 0.05

    def uniform_random(self):
        sample = np.random.uniform(low=-1, high=1, size=2)
        return sample

    def dist(self, state1, state2):
        state1 = np.array(state1)
        state2 = np.array(state2)
        distance = (state1 - state2) @ np.transpose(state1 - state2)
        return np.sqrt(np.sum(distance))

    def near_goal(self, state):
        if self.dist(state, self.goal.position) < self.delta:
            return True
        return False

    def grow_tree(self):
        x_rand = self.uniform_random()

        father = 0
        min_dist = self.dist(x_rand, self.tree[0].position)
        for i in range(1, len(self.tree)):
            dist = self.dist(x_rand, self.tree[i].position)

            if dist <= min_dist:
                min_dist = dist
                father = i

        x_near = self.tree[father].position

        if self.dist(x_near, x_rand) < self.upper:
            position = x_rand
        else:
            k = (x_rand[1] - x_near[1]) / (x_rand[0] - x_near[0])
            delta_x = self.upper / np.sqrt((1 + k ** 2))

            if x_rand[0] < 0:
                delta_x = -1 * delta_x

            position = [x_near[0] + delta_x, x_near[1] + delta_x * k]

        x_new = MyLeaf(0, father, position)
        self.tree.append(x_new)

    def plot_tree(self, radius=1):
        state = np.array(self.goal.position)
        circle = patches.Circle((state + 1.0), radius=0.01 * radius, facecolor='r', edgecolor='r')
        plt.gca().add_patch(circle)

        plt.axis([0.0, 2.0, 0.0, 2.0])
        for i in range(len(self.tree)):
            state = np.array(self.tree[i].position)
            circle = patches.Circle((state + 1.0), radius=0.01 * radius)
            plt.gca().add_patch(circle)

            next_state = np.array(self.tree[self.tree[i].father].position)
            path = patches.ConnectionPatch(state + 1.0, next_state + 1.0, 'data', arrowstyle='-',
                                           color='red')
            plt.gca().add_patch(path)
            plt.pause(0.1)
            plt.ioff()

    def find_goal(self, m):
        for _ in range(m):
            self.grow_tree()

            if self.near_goal(self.tree[-1].position):
                print('Find goal!')
                self.goal.father = len(self.tree)-1
                self.goal.root = 0
                self.tree.append(self.goal)
                break


if __name__ == '__main__':
    rrt = MyRRT()

    rrt.find_goal(500)

    rrt.plot_tree()
    plt.show()
