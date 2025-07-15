import math
import random
import matplotlib.pyplot as plt
import numpy as np


class RRT:
    class Node:  # 创建节点
        def __init__(self, x, y):
            self.x = x  # 节点坐标
            self.y = y
            self.path_x = []  # 路径，作为画图的数据
            self.path_y = []
            self.parent = None  # 父节点

    class AreaBounds:
        """区域大小
        """

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=1.0,  # 树枝长度
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter
        start:起点 [x,y]
        goal:目标点 [x,y]
        obstacleList:障碍物位置列表 [[x,y,size],...]
        rand_area: 采样区域 x,y ∈ [min,max]
        play_area: 约束随机树的范围 [xmin,xmax,ymin,ymax]
        robot_radius: 机器人半径
        expand_dis: 扩展的步长
        goal_sample_rate: 采样目标点的概率，百分制.default: 5，即表示5%的概率直接采样目标点
        """
        self.start = self.Node(start[0], start[1])  # 根节点(0,0)
        self.end = self.Node(goal[0], goal[1])  # 终点(6,10)
        self.min_rand = rand_area[0]  # -2  树枝生长区域xmin
        self.max_rand = rand_area[1]  # 15  xmax

        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)  # 树枝生长区域，左下(-2,0)==>右上(12,14)
        else:
            self.play_area = None  # 数值无限生长

        self.expand_dis = expand_dis  # 树枝一次的生长长度
        self.goal_sample_rate = goal_sample_rate  # 多少概率直接选终点
        self.max_iter = max_iter  # 最大迭代次数
        self.obstacle_list = obstacle_list  # 障碍物的坐标和半径
        self.node_list = []  # 保存节点
        self.robot_radius = robot_radius  # 随机点的搜索半径

    # 路径规划
    def planning(self, animation=True, camara=None):

        # 将起点作为根节点x_{init}​，加入到随机树的节点集合中。
        self.node_list = [self.start]  # 先在节点列表中保存起点
        for i in range(self.max_iter):
            # 从可行区域内随机选取一个节点x_{rand}
            rnd_node = self.sample_free()

            # 已生成的树中利用欧氏距离判断距离x_{rand}​最近的点x_{near}。
            # 从已知节点中选择和目标节点最近的节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)  # 最接近的节点的索引
            nearest_node = self.node_list[nearest_ind]  # 获取该最近已知节点的坐标

            # 从 x_{near} 与 x_{rand} 的连线方向上扩展固定步长 u，得到新节点 x_{new}
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 如果在可行区域内，且x_{near}与x_{new}之间无障碍物
            # 判断新点是否在规定的树的生长区域内，新点和最近点之间是否存在障碍物
            if self.is_inside_play_area(new_node, self.play_area) and \
                    self.obstacle_free(new_node, self.obstacle_list, self.robot_radius):
                # 都满足才保存该点作为树节点
                self.node_list.append(new_node)

            # 如果此时得到的节点x_new到目标点的距离小于扩展步长，则直接将目标点作为x_rand。
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                # 以新点为起点，向终点画树枝
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                # 如果最新点和终点之间没有障碍物True
                if self.obstacle_free(final_node, self.obstacle_list, self.robot_radius):
                    # 返回最终路径
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation:
                self.draw_graph(rnd_node, camara)

        return None  # cannot find path

    # 距离最近的已知节点坐标，随机坐标，从已知节点向随机节点的延展的长度
    def steer(self, from_node, to_node, extend_length=float("inf")):
        # d已知点和随机点之间的距离，theta两个点之间的夹角
        d, theta = self.calc_distance_and_angle(from_node, to_node)

        # 如果$x_{near}$与$x_{rand}$间的距离小于步长，则直接将$x_{rand}$作为新节点$x_{new}$
        if extend_length >= d:  # 如果树枝的生长长度超出了随机点，就用随机点位置作为新节点
            new_x = to_node.x
            new_y = to_node.y
        else:  # 如果树生长长度没达到随机点长度，就截取长度为extend_length的节点作为新节点
            new_x = from_node.x + math.cos(theta) * extend_length  # 最近点 x + cos * extend_len
            new_y = from_node.y + math.sin(theta) * extend_length  # 最近点 y + sin * extend_len

        new_node = self.Node(new_x, new_y)  # 初始化新节点
        new_node.path_x = [from_node.x]  # 最近点
        new_node.path_y = [from_node.y]  #
        new_node.path_x.append(new_x)  # 新点
        new_node.path_y.append(new_y)

        new_node.parent = from_node  # 根节点变成最近点，用来指明方向

        return new_node

    def generate_final_course(self, goal_ind):  # 终点坐标的索引
        """生成路径
        Args:
            goal_ind (_type_): 目标点索引
        Returns:
            _type_: _description_
        """
        path = [[self.end.x, self.end.y]]  # 保存终点节点
        node = self.node_list[goal_ind]
        while node.parent is not None:  # 根节点
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        """计算(x,y)离目标点的距离
        """
        dx = x - self.end.x  # 新点的x-终点的x
        dy = y - self.end.y
        return math.hypot(dx, dy)  # 计算新点和终点之间的距离

    def sample_free(self):
        # 以（100-goal_sample_rate）%的概率随机生长，(goal_sample_rate)%的概率朝向目标点生长
        if random.randint(0, 100) > self.goal_sample_rate:  # 大于5%就不选终点方向作为下一个节点
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),  # 在树枝生长区域中随便取一个点
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    # 绘制搜索过程
    def draw_graph(self, rnd=None, camera=None):
        if camera == None:
            plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # 画随机点
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        # 画已生成的树
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # 画障碍物
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        # 如果约定了可行区域，则画出可行区域
        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        # 画出起点和目标点
        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.1)
        if camera != None:
            camera.snap()

    # 静态方法无需实例化，也可以实例化后调用，静态方法内部不能调用self.的变量
    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):  # 已知节点list，随机的节点坐标
        # 计算所有已知节点和随机节点之间的距离
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        # 获得距离最小的节点的索引
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def is_inside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def obstacle_free(node, obstacleList, robot_radius):  # 目标点，障碍物中点和半径，移动时的占地半径

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """计算两个节点间的距离和方位角
        Args:
            from_node (_type_): _description_
            to_node (_type_): _description_
        Returns:
            _type_: _description_
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)  # 平方根
        theta = math.atan2(dy, dx)  # 夹角的弧度值
        return d, theta


def main(gx=6.0, gy=10.0):
    print("start " + __file__)
    fig = plt.figure(1)

    # camera = Camera(fig) # 保存动图时使用
    camera = None  # 不保存动图时，camara为None
    show_animation = True
    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],  # 起点位置
        goal=[gx, gy],  # 终点位置
        rand_area=[-2, 15],  # 树枝可生长区域[xmin,xmax]
        obstacle_list=obstacleList,  # 障碍物
        play_area=[-2, 12, 0, 14],  # 树的生长区域，左下[-2,0]==>右上[12,14]
        robot_radius=0.2  # 搜索半径
    )
    path = rrt.planning(animation=show_animation, camara=camera)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # 绘制最终路径
        if show_animation:
            rrt.draw_graph(camera=camera)
            plt.grid(True)
            plt.pause(0.1)
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            if camera != None:
                camera.snap()
                # animation = camera.animate()
                # animation.save('trajectory.gif')
            plt.show()


if __name__ == '__main__':
    main()
