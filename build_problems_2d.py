import numpy as np
from numpy import savez_compressed
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ProblemMaker:
    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.delta_x = 0
        self.delta_y = 0
        self.map = []
        self.size = 0

    def build_map(self, size=15):
        self.size = size
        self.map = np.zeros([self.size, self.size])

        self.map[:, 0] = 1
        self.map[:, -1] = 1
        self.map[0, :] = 1
        self.map[-1, :] = 1
        for _ in range(np.random.randint(3, 1000, 1)[0]):
            self.create_obs()

        return self.map

    def create_obs(self):
        while True:
            self.pos_x, self.pos_y = np.random.randint(2, self.size - 2, 1)[0], np.random.randint(2, self.size - 2, 1)[
                0]
            sig = self.map[self.pos_x + 1][self.pos_y + 1] + self.map[self.pos_x + 1][self.pos_y - 1] + \
                  self.map[self.pos_x - 1][self.pos_y + 1] + self.map[self.pos_x - 1][self.pos_y - 1]
            if sig == 0 or self.map[self.pos_x][self.pos_y] == 1:
                break
        dire = np.random.randint(0, 4, 1)[0]
        length = np.random.randint(3, self.size - 1, 1)[0]

        if dire == 0:
            self.delta_x = 1
        elif dire == 1:
            self.delta_x = -1
        elif dire == 2:
            self.delta_y = 1
        elif dire == 3:
            self.delta_y = -1

        self.map[self.pos_x][self.pos_y] = 1
        for i in range(length):
            self.pos_x = self.pos_x + self.delta_x
            self.pos_y = self.pos_y + self.delta_y
            if self.check_connected():

                sig = self.map[self.pos_x + 1][self.pos_y + 1] + self.map[self.pos_x + 1][self.pos_y - 1] + \
                      self.map[self.pos_x - 1][self.pos_y + 1] + self.map[self.pos_x - 1][self.pos_y - 1]
                if sig != 0 and 1 < self.pos_x < length - 2 and 1 < self.pos_y < length - 2:
                    self.map[self.pos_x - self.delta_x][self.pos_y - self.delta_y] = 0
                    break

                self.map[self.pos_x][self.pos_y] = 1

            else:
                break
        self.delta_y = 0
        self.delta_x = 0

    def check_connected(self):
        if 0 < self.pos_x < self.size - 1 and 0 < self.pos_y < self.size - 1:
            sig = self.map[self.pos_x + 1][self.pos_y] + self.map[self.pos_x][self.pos_y + 1] + \
                  self.map[self.pos_x - 1][self.pos_y] + self.map[self.pos_x][self.pos_y - 1]

            near_edge = self.pos_x == 1 or self.pos_y == 1

            if sig <= 1:
                return True
            else:
                return False

        self.map[self.pos_x][self.pos_y] = 1
        return False

    def check_point(self, state, map_width=15):
        '''
            将[-1,1]的state值（double值）转成map中的坐标（int）,然后返回该state是否在障碍物中,1为不在，0为在(LIMITE==1)
        '''
        coordinate = ((state + 1) * map_width / 2).astype(int)
        return self.map[coordinate[0]][coordinate[1]] == 0  # 还需要判断state的合理性

    def sample_point(self):
        sample = np.random.uniform(low=-1, high=1, size=2)
        return sample

    def plot_map(self):
        rect = patches.Rectangle((0.0, 0.0), 2.0, 2.0, linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        map = self.map

        map_width = self.size
        d_x = 2.0 / map_width
        d_y = 2.0 / map_width

        for i in range(map_width):
            for j in range(map_width):
                if map[i][j] > 0:
                    rect = patches.Rectangle((d_x * i, d_y * j), d_x, d_y, linewidth=1, edgecolor='#253494',
                                             facecolor='#253494')
                    plt.gca().add_patch(rect)

        plt.axis([0.0, 2.0, 0.0, 2.0])
        # plt.subplots_adjust(left=-0., right=1.0, top=1.0, bottom=-0.)
        return plt

    def plot_nodes(self, state, title=' ', radius=1, color='#e6550d'):
        circle = patches.Circle((state + 1.0), radius=0.01 * radius, edgecolor=color, facecolor=color)
        plt.gca().add_patch(circle)
        plt.annotate(title, state + 1, color='black', backgroundcolor=(1., 1., 1., 0.), fontsize=15)


if __name__ == '__main__':
    p = ProblemMaker()
    size = 3000
    ms = []
    init_states = []
    goal_states = []

    # with np.load(map_file) as f:
    #     ps = f['maps']
    #     print(1)

    for i in range(size):
        m = p.build_map(1000)

        while True:
            init_state = p.sample_point()
            if p.check_point(init_state):
                break

        while True:
            goal_state = p.sample_point()
            if p.check_point(goal_state):
                break

        a = p.plot_map()
        # print(m)
        p.plot_nodes(init_state, title='init_state')
        p.plot_nodes(goal_state, title='goal_state')
        a.show()

        ms.append(m)
        goal_states.append(goal_state)
        init_states.append(init_state)

    problems = {'maps': ms, 'goal_states': goal_states, 'init_states': init_states}

    f = open('maze_files/train_2d.pkl', 'wb')
    pickle.dump(problems, f)
    f.close()

    a = np.load('maze_files/train_2d.pkl', encoding='bytes', allow_pickle=True)

    print(a)
