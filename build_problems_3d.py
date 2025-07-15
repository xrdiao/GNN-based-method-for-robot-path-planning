import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import joblib
import pickle
from tqdm import tqdm
import time


class ProblemMaker3d:
    def __init__(self):
        self.problems = []
        self.model = None
        self.obss_info = []

        self.pose_range = []
        self.bound = []
        self.config_dim = 0

    def check_point(self, model, state):
        '''
        Returns:
            True - 没有碰撞
            False - 发生碰撞
        '''
        for i in range(p.getNumJoints(model)):
            p.resetJointState(model, i, state[i])

        p.performCollisionDetection()
        p.stepSimulation()

        if p.getContactPoints(model):  # p.getContact 碰撞是True, 否则是False
            return False
        else:
            return True

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
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=obs_collision_id,
                          baseVisualShapeIndex=obs_shape_id,
                          basePosition=position)

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

    def generate_problem(self, GUI=False):
        sizes = np.random.randint(2, 5, 1)

        if GUI:
            p.connect(p.GUI,
                      options='--background_color_red=0.9 --background_color_green=0.9 --background_color_blue=0.9')
        else:
            p.connect(p.DIRECT)
        self.model = p.loadURDF('ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf',
                                [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

        self.pose_range = [(p.getJointInfo(self.model, jointId)[8], p.getJointInfo(self.model, jointId)[9]) for
                           jointId in
                           range(p.getNumJoints(self.model))]
        self.bound = np.array(self.pose_range).T.reshape(-1)
        self.config_dim = p.getNumJoints(self.model)

        for _ in np.arange(sizes):
            size = np.random.uniform(0.1, 0.3, 3)
            position = np.hstack([np.random.uniform(-1, 1, 2), np.random.uniform(0.75, 1.5, 1)])

            if self.check_point(self.model, [0, 0, 0, 0, 0, 0, 0]):
                obs_info = tuple([size, position])
                self.obss_info.append(obs_info)
                self.create_obs(size, position)

    def build_problems(self, n):
        for _ in tqdm(range(n)):
            problem = []
            self.obss_info = []

            self.generate_problem()
            model = self.model

            check_map = 0
            while True:
                # init_state = np.random.uniform(low=np.array(self.pose_range)[:, 0],
                #                                high=np.array(self.pose_range)[:, 1],
                #                                size=self.config_dim)
                init_state = np.random.uniform(low=-2.65, high=2.65, size=7)
                for i in range(p.getNumJoints(model)):
                    p.resetJointState(model, i, init_state[i])
                a = p.getLinkState(model, p.getNumJoints(model) - 1)[0]

                if self.check_point(model, init_state):
                    break
                check_map = check_map + 1

                if check_map == 200:
                    p.disconnect()
                    check_map = 0
                    self.generate_problem()

            while True:
                # goal_state = np.random.uniform(low=np.array(self.pose_range)[:, 0],
                #                                high=np.array(self.pose_range)[:, 1],
                #                                size=self.config_dim)
                goal_state = np.random.uniform(low=-2.65, high=2.65, size=7)
                for i in range(p.getNumJoints(model)):
                    p.resetJointState(model, i, goal_state[i])
                a = p.getLinkState(model, p.getNumJoints(model) - 1)[0]
                if self.check_point(model, goal_state):
                    break

            problem.append(self.obss_info)

            problem.append(init_state)
            problem.append(goal_state)

            # self.check_sim(init_state)
            # time.sleep(3)
            #
            # self.check_sim(goal_state)
            # time.sleep(3)
            problem.append([0])
            self.problems.append(problem)
            p.disconnect()

        return self.problems


if __name__ == '__main__':
    pm = ProblemMaker3d()
    problems = pm.build_problems(50)

    f = open('maze_files/test_6.pkl', 'wb')
    pickle.dump(problems, f)
    f.close()

    # problems = np.load('maze_files/test.pkl', encoding='bytes', allow_pickle=True)

    print(1)
