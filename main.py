from Dijkstra import Dijkstra
from map_3d import maps_3d
import numpy as np
import time
from tqdm import tqdm
from my_graph import my_graph
import torch
from MyGNN import Guidance
from MyMotionPlan import MyMotionPlan
import BIT
from MyRRT_star import MyRRT_star

LIMITE = 1.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 记得改


def cal_var(data, epoch):
    avg_col = np.sum(data, axis=0) / epoch
    var_col = np.array(data) - avg_col
    var_col = np.sqrt(var_col @ var_col.T) / epoch
    return avg_col, var_col


def dijkstra_plan(map_index, n=200, map_file='maze_files/kukas_7_3000.pkl', model_file='kuka_iiwa/model_0.urdf'):
    print('----------- Dijkstra -------------')
    epoch = len(map_index)
    m = maps_3d(map_file=map_file, model_file=model_file)
    p_len = []
    success_rate = 0

    d = Dijkstra(m)
    times = []
    t_plan = []
    collisions = []
    t_sam = []
    k = n / 4

    for i in tqdm(range(epoch)):
        t_1 = time.time()
        choice = map_index[i]
        m.init_problem(choice)

        # 采样
        nodes, _ = m.sample_points(n)
        if nodes is None:
            print(map_index[i])
            continue
        m.nodes = nodes
        init_index = np.where(m.nodes == m.init_state)[0][0]
        goal_index = np.where(m.nodes == m.goal_state)[0][0]

        # 构建图
        m.connect_edge(nodes, k=k)
        t_sam.append(time.time() - t_1)

        # 规划
        time_start = time.time()
        d.dijkstra(init_index, goal_index)
        a = d.get_path(goal_index)
        time_end = time.time()
        t_plan.append(time_end - time_start)

        if a is not None and a[0] == m.goal_index:
            success_rate += 1
            length = 0
            for j in range(len(a) - 1):
                length = length + m.dist(nodes[j], nodes[j + 1])
                p_len.append(length)

        t_2 = time.time()
        times.append(t_2 - t_1)

        collisions.append(d.obstacle_times)
        d.obstacle_times = 0

    # 规划时间
    avg_plan, var_plan = cal_var(t_plan, epoch)

    # 整体时间
    avg_tot, var_tot = cal_var(times, epoch)

    # 碰撞次数
    avg_col, var_col = cal_var(collisions, epoch)

    avg_len, var_len = cal_var(p_len, epoch)

    success_rate = success_rate / epoch * 100

    print('epoch:', epoch)
    print('avg collision:', avg_col)
    print('avg length of path:', avg_len)
    print('avg planning time:', avg_plan)
    print('avg total time:', avg_tot)
    print('var tot:', var_tot)
    print('var col:', var_col)
    print('var plan:', var_plan)
    print('success rate:', success_rate, '%')

    return [avg_col, var_col, avg_tot, var_tot, avg_plan, var_plan, avg_len, var_len, success_rate]


def BIT_plan(map_index, iter_max=200, map_file='maze_files/kukas_7_3000.pkl', model_file='kuka_iiwa/model_0.urdf'):
    print('----------- BIT -------------')
    epoch = len(map_index)
    m = maps_3d(map_file=map_file, model_file=model_file)
    p_len = []
    success_rate = 0
    eta = 2
    bit = BIT.BITStar(eta, iter_max, m)
    times = []
    t_plan = []
    collisions = []
    t_sam = []

    for i in tqdm(range(epoch)):
        t_1 = time.time()
        choice = map_index[i]
        m.init_problem(choice)

        # 规划
        time_start = time.time()
        bit.planning()
        path = bit.ExtractPath()
        time_end = time.time()
        t_plan.append(time_end - time_start)

        if path[0].any() == m.goal_state.any() and len(path) > 1:
            success_rate += 1
            length = 0
            for j in range(len(path) - 1):
                length = length + m.dist(path[j], path[j + 1])
                p_len.append(length)

        t_2 = time.time()
        times.append(t_2 - t_1)

        collisions.append(m.obstacle_times)
        m.obstacle_times = 0

    # 规划时间
    avg_plan, var_plan = cal_var(t_plan, epoch)

    # 整体时间
    avg_tot, var_tot = cal_var(times, epoch)

    # 碰撞次数
    avg_col, var_col = cal_var(collisions, epoch)

    avg_len, var_len = cal_var(p_len, epoch)

    success_rate = success_rate / epoch * 100

    print('epoch:', epoch)
    print('avg collision:', avg_col)
    print('avg length of path:', avg_len)
    print('avg planning time:', avg_plan)
    print('avg total time:', avg_tot)
    print('var tot:', var_tot)
    print('var col:', var_col)
    print('var plan:', var_plan)
    print('success rate:', success_rate, '%')

    return [avg_col, var_col, avg_tot, var_tot, avg_plan, var_plan, avg_len, var_len, success_rate]


def RRT_plan(map_index, iter_max=200, map_file='maze_files/kukas_7_3000.pkl', model_file='maze_files/kukas_7_3000.pkl'):
    print('----------- RRT -------------')
    epoch = len(map_index)
    m = maps_3d(map_file=map_file, model_file=model_file)
    p_len = []
    success_rate = 0
    rrt = MyRRT_star(m)
    times = []
    t_plan = []
    collisions = []

    for i in tqdm(range(epoch)):
        t_1 = time.time()
        choice = map_index[i]
        m.init_problem(choice)

        # 规划
        time_start = time.time()
        rrt.find_goal(iter_max)
        path = rrt.get_path()
        time_end = time.time()
        t_plan.append(time_end - time_start)

        if path[0].any() == m.goal_state.any() and len(path) > 1:
            success_rate += 1
            length = 0
            for j in range(len(path) - 1):
                length = length + m.dist(path[j], path[j + 1])
                p_len.append(length)

        t_2 = time.time()
        times.append(t_2 - t_1)

        collisions.append(m.obstacle_times)
        m.obstacle_times = 0

    # 规划时间
    avg_plan, var_plan = cal_var(t_plan, epoch)

    # 整体时间
    avg_tot, var_tot = cal_var(times, epoch)

    # 碰撞次数
    avg_col, var_col = cal_var(collisions, epoch)

    avg_len, var_len = cal_var(p_len, epoch)

    success_rate = success_rate / epoch * 100

    print('epoch:', epoch)
    print('avg collision:', avg_col)
    print('avg length of path:', avg_len)
    print('avg planning time:', avg_plan)
    print('avg total time:', avg_tot)
    print('var tot:', var_tot)
    print('var col:', var_col)
    print('var plan:', var_plan)
    print('success rate:', success_rate, '%')

    return [avg_col, var_col, avg_tot, var_tot, avg_plan, var_plan, avg_len, var_len, success_rate]


def model_plan(map_index, n=200, map_file='maze_files/kukas_7_3000.pkl'):
    print('----------- model -------------')
    epoch = len(map_index)
    # maps = torch.load("data/map.pth")
    m = maps_3d(map_file=map_file)

    # 超参量
    success_rate = 0
    p_len = []
    times = []
    t_sam = []
    t_plan = []
    collisions = []

    g = Guidance(node_channel=14, map_chanel=1, obs_channel=6, out_channel=32, batch=1, init_net=False)
    g.load_state_dict(torch.load('model/model.pth'))
    g.to(device)
    g.eval()

    p = MyMotionPlan(m)
    my_gra = my_graph(m)
    k = n / 4

    # 开始运行
    for j in tqdm(range(epoch)):
        t_1 = time.time()
        ind = map_index[j]
        m.init_problem(ind)
        g.env_coder = None

        t_s = time.time()
        nodes, _ = m.sample_points(n)
        if nodes is None:
            print(map_index[j])
            continue
        m.nodes = nodes
        m.map = torch.tensor(m.build_map(15).reshape(1, 15, 15, 15), dtype=torch.float)
        m.connect_edge2(m.nodes, k)
        t_e = time.time()
        t_sam.append(t_e - t_s)

        # goal_index = np.where(m.nodes == m.goal_state[0])[0][0]
        data = torch.cat(
            (torch.tensor(np.array(nodes), dtype=torch.float),
             torch.tensor(m.goal_state, dtype=torch.float).unsqueeze(0).repeat(len(nodes), 1)), dim=1)
        edge_index = my_gra.matrix2list(m.graph)

        # 规划
        t_s = time.time()
        path_guidance = p.plan(len(nodes), g, data, edge_index, m.map)  # 搜索的路径节点原来设置为100个点
        t_d = time.time()
        t_plan.append(t_d - t_s)

        if path_guidance[-1] == m.goal_index:
            success_rate += 1
            for i in range(len(path_guidance) - 1):
                p_len.append(m.dist(nodes[i], nodes[i + 1]))

        t_2 = time.time()
        times.append(t_2 - t_1)

        collisions.append(p.obstacle_times)
        p.obstacle_times = 0

    # 规划时间
    avg_plan, var_plan = cal_var(t_plan, epoch)

    # 整体时间
    avg_tot, var_tot = cal_var(times, epoch)

    # 碰撞次数
    avg_col, var_col = cal_var(collisions, epoch)

    avg_len, var_len = cal_var(p_len, epoch)

    success_rate = success_rate / epoch * 100

    print('epoch:', epoch)
    print('avg collision:', avg_col)
    print('avg length of path:', avg_len)
    print('avg planning time:', avg_plan)
    print('avg total time:', avg_tot)
    print('var tot:', var_tot)
    print('var col:', var_col)
    print('var plan:', var_plan)
    print('success rate:', success_rate, '%')

    return [avg_col, var_col, avg_tot, var_tot, avg_plan, var_plan, avg_len, var_len, success_rate]


def model_plan_6(map_index, n=200, map_file='maze_files/test.pkl'):
    print('----------- model -------------')
    epoch = len(map_index)
    m = maps_3d(GUI=False,
                model_file='ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf',
                map_file=map_file)

    # 超参量
    success_rate = 0
    p_len = []
    times = []
    t_sam = []
    t_plan = []
    collisions = []

    g = Guidance(node_channel=14, map_chanel=1, obs_channel=6, out_channel=32, batch=1, init_net=False)
    g.load_state_dict(torch.load('model/model.pth'))
    g.to(device)
    g.eval()

    p = MyMotionPlan(m)
    my_gra = my_graph(m)
    k = n / 4

    # 开始运行
    for j in tqdm(range(epoch)):
        t_1 = time.time()
        ind = map_index[j]
        m.init_problem(ind)
        g.env_coder = None

        t_s = time.time()
        nodes, _ = m.sample_points(n)
        if nodes is None:
            print(map_index[j])
            continue
        m.nodes = nodes
        m.map = torch.tensor(m.build_map(15).reshape(1, 15, 15, 15), dtype=torch.float)
        m.connect_edge2(m.nodes, k)
        t_e = time.time()
        t_sam.append(t_e - t_s)

        data = torch.cat(
            (torch.tensor(np.array(nodes), dtype=torch.float),
             torch.tensor(m.goal_state, dtype=torch.float).unsqueeze(0).repeat(len(nodes), 1)), dim=1)
        edge_index = my_gra.matrix2list(m.graph)

        # 规划
        t_s = time.time()
        path_guidance = p.plan(len(nodes), g, data, edge_index, m.map)  # 搜索的路径节点原来设置为100个点
        t_d = time.time()
        t_plan.append(t_d - t_s)

        if path_guidance[-1] == m.goal_index:
            success_rate += 1
            for i in range(len(path_guidance) - 1):
                p_len.append(m.dist(nodes[i], nodes[i + 1]))

        t_2 = time.time()
        times.append(t_2 - t_1)

        collisions.append(p.obstacle_times)
        p.obstacle_times = 0

    # 规划时间
    avg_plan, var_plan = cal_var(t_plan, epoch)

    # 整体时间
    avg_tot, var_tot = cal_var(times, epoch)

    # 碰撞次数
    avg_col, var_col = cal_var(collisions, epoch)

    avg_len, var_len = cal_var(p_len, epoch)

    success_rate = success_rate / epoch * 100

    print('epoch:', epoch)
    print('avg collision:', avg_col)
    print('avg length of path:', avg_len)
    print('avg planning time:', avg_plan)
    print('avg total time:', avg_tot)
    print('var tot:', var_tot)
    print('var col:', var_col)
    print('var plan:', var_plan)
    print('success rate:', success_rate, '%')

    return [avg_col, var_col, avg_tot, var_tot, avg_plan, var_plan, avg_len, var_len, success_rate]


if __name__ == '__main__':
    size = 300
    set1 = np.arange(50)
    sampling = [400]

    # data_bit_6 = [BIT_plan(set1, sampling[i], map_file='maze_files/test.pkl',
    # model_file='ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf')
    # for i in range(len(sampling))]
    data_model_6 = [model_plan_6(set1, sampling[i]) for i in range(len(sampling))]
    # data_model = [model_plan(set1, sampling[i], map_file='maze_files/test.pkl') for i in range(len(sampling))]
    # data_rrt_6 = [RRT_plan(set1, sampling[i], map_file='maze_files/test.pkl',
    #                        model_file='ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf')
    #               for i in range(len(sampling))]
    # data_prm_6 = [dijkstra_plan(set1, sampling[i], map_file='maze_files/test.pkl',
    #                             model_file='ros_kortex-noetic-devel/kortex_description/arms/gen3_lite/6dof/urdf/GEN3-LITE.urdf')
    #               for i in range(len(sampling))]
    #
    # for i in range(len(sampling)):
    #     BIT_plan(set1, sampling[i], map_file='maze_files/test.pkl')
    #     model_plan_6(set1, sampling[i])

    # np.save('data/data_bit_6', data_bit_6)
    # np.save('data/data_model_6', data_model_6)
    # np.save('data/data_rrt_6', data_rrt_6)
    # np.save('data/data_prm_6', data_prm_6)

    # a = np.load('data/data_model.npy')
    # print(a)
