import numpy as np
import json
from Tasks.ro_con import StripPathGenerator
from Tasks.t import get_sub_area, get_all_path,choose_path, get_distance
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from Tasks.config import get_config
from Tasks.dubins import dubins_path_planning
from Tasks.plot import plot_turn, generate_map, plot_initial
# from Tasks.tests import ga_orders
# from test import ga_order
pi = 3.1415926

color = []
for c in colors.cnames:
    color.append(c)
def num_e(l,target):
    b = []
    for index, nums in enumerate(l):
        if nums == target:
            b.append(index)
    return b

def com_path(sub_path):
    # print(sub_path)
    com_path = sub_path[-2] + sub_path[-1]
    # print(com_path, sub_path[-2], sub_path[-1])
    sub_path.pop(-1)
    # print(sub_path)
    sub_path.pop(-1)
    # print(sub_path)
    sub_path.append(com_path)
    return sub_path

def get_yaw(end_pos,star_pos):
    theta = np.arctan((end_pos[1]-star_pos[1])/(end_pos[0]-star_pos[0])) if star_pos[0] != end_pos[0] else -pi/2# -pi
    if end_pos[0] < star_pos[0]:  # 在二三象限
        theta = theta+pi
    end_yaw = theta
    return end_yaw

# 获得对各个区域的覆盖航路
def get_cover_path(area_all,lx):
    area_num = len(area_all)
    area_task = [[] for _ in range(area_num)]  # area_num 行 4 列[[], [], [], [], []]
    yaw_task = [[] for _ in range(area_num)]  # area_num 行 4 列
    path_all = []
    # 分别对各个区域进行航路规划
    for i in range(0, len(area_all)):
        # 获得当前区域的信息点
        convhull_point = area_all[i]
        # 获得旋转后的凸包，一个方向的航道，航迹点，旋转后的航道
        ro_convhull_point, path, node_pos, patht = StripPathGenerator(convhull_point, lx)  # 从奇数点开始拐弯
        # 获得所有方向的航迹点
        node_all = get_all_path(node_pos)
        # 获得所有区域所有方向的航道
        path_all.append(node_all)
        # 获得当前区域，四种搜索方向的进入点和离开点
        for j in range(4):
            area_task[i].append([node_all[j][0], node_all[j][-1]])
            entry_angle = get_yaw(node_all[j][1], node_all[j][0])
            end_yaw = entry_angle if np.shape(node_all[j])[0]/2 % 2 == 1 else entry_angle+pi
            yaw_task[i].append([entry_angle, end_yaw])
    return path_all,area_task,yaw_task

def greedy(map,ax1,area_num,start_city,path_all,curvature):
    # 底图上画上区域以及无人机的出发区域
    map.map_plot(ax1)
    ax1.plot(start_city[0], start_city[1], 'ks', markersize=10)  # 无人机的起点, label='initial'
    distance_all1 = []  # 存储几个区域的路程
    s_all1 = []
    path_case1 = []
    dis_length1 = 0
    greedy_order = []
    co=['-y','-m','-k']
    path_all_copy = path_all.copy()
    start_position= start_city.copy()
    for i in range(0, area_num):
        # 根据当前位置选择一个最近的航道,绘制到达区域的路径，path_all_copy会被改变，删去已经选择的区域
        # node_pos： 选择的区域的航迹点；path_num表示所选区域的极其方向的索引
        node_pos, path_num = choose_path(path_all_copy, start_position, curvature)
        greedy_order.append(path_num)
        # print(greedy_order)
        path_case1.append(node_pos)
        # 计算距离
        path_length = plot_initial(start_position, node_pos, curvature, ax1, co[0])
        dis_length1 = dis_length1 + path_length
        # 到达搜索区域，开始搜索,计算全程的距离
        x = np.array(node_pos)[:, 0]
        y = np.array(node_pos)[:, 1]
        node_index = np.array(node_pos)
        start_yaw = plot_turn(node_index, curvature, 'b', ax1)
        ax1.scatter(x[0], y[0], s=80, c='k')
        ax1.scatter(x[-1], y[-1], s=80, c='k')
        ax1.scatter(x[0], y[0], s=20, c='r')
        ax1.scatter(x[-1], y[-1], s=20, c='r')
        # ax1.plot(x[0], y[0], 'r*', markersize=10, label='start')
        # ax1.plot(x[-1], y[-1], 'rx', markersize=10, label='end')
        distance, s = get_distance(node_pos, curvature)
        # 存储路径的总路程信息和航路点距离信息
        s_all1.append(s)
        distance_all1.append(distance)
        # 搜索完后，无人机的位置需要更新
        start_position = [x[-1], y[-1], start_yaw]
    # 返回原点
    path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(start_position[0], start_position[1], start_position[2], start_city[0],
                                                                       start_city[1], pi, curvature)
    # ax1.plot(path_x0, path_y0, '-')  # 到达任务区域的航迹点, 'm--'
    dis_length1 = dis_length1 + path_length
    ax1.title.set_text("Greedy Algorithm——" + 'Total distance=' + str(np.round(dis_length1)))
    ax1.set_aspect(1)    # plt.legend()
    ax1.grid()
    return s_all1, path_case1,distance_all1

def ga_tsp(map,ax2,start_city,path_all,curvature,order):
    co = ['-y', '-m', '-k']
    order0 = [int(i) for i in order]
    order1 = [int(i*10 % 10) for i in order]
    map.map_plot(ax2)
    ax2.plot(start_city[0], start_city[1], 'ks', markersize=10)  # 无人机的起点, label='initial'
    distance_all2 = []  # 存储几个区域的路程
    s_all2 = []
    path_case2 = []
    dis_length2 = 0
    start_position_copy = start_city.copy()
    for index, path_index in zip(order0, order1):
        # 根据当前位置选择一个最近的航道,绘制到达区域的路径
        node_pos = path_all[index][path_index]
        path_case2.append(node_pos)
        path_length = plot_initial(start_position_copy, node_pos, curvature, ax2, co[0])
        dis_length2 = dis_length2 + path_length
        # 到达搜索区域，开始搜索,计算全程的距离
        x = np.array(node_pos)[:, 0]
        y = np.array(node_pos)[:, 1]
        node_index = np.array(node_pos)
        start_yaw = plot_turn(node_index, curvature, 'b', ax2)
        ax2.scatter(x[0], y[0], s=80, c='k')
        ax2.scatter(x[-1], y[-1], s=80, c='k')
        ax2.scatter(x[0], y[0], s=20, c='r')
        ax2.scatter(x[-1], y[-1], s=20, c='r')
        # ax2.plot(x[0], y[0], 'r*', markersize=10, label='start')
        # ax2.plot(x[-1], y[-1], 'rx', markersize=10, label='end')
        distance, s = get_distance(node_pos, curvature)
        # 存储路径的总路程信息和航路点距离信息
        s_all2.append(s)
        distance_all2.append(distance)
        # 搜索完后，无人机的位置需要更新
        start_position_copy = [x[-1], y[-1], start_yaw]
    path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(start_position_copy[0], start_position_copy[1],start_position_copy[2], start_city[0],
                                                                       start_city[1], pi, curvature)
    # ax2.plot(path_x0, path_y0, '-')  # 到达任务区域的航迹点, 'm--'
    dis_length2 = dis_length2 + path_length
    # print(dis_length2, sum(s_all2))
    # 绘图
    # ax2.title.set_text("Genetic Algorithm——"+'Total distance='+str(dis_length2))
    ax2.set_aspect(1)    # plt.legend()
    ax2.grid()
    # plt.show()

def area_cut(map,ax3,s_all1,path_case1,distance_all1,curvature,area_num):
    map.map_plot(ax3)
    # 以最小路程为区域任务的分割标准
    ind_min = np.argmin(s_all1)
    min_s = s_all1[ind_min]
    s_task = []
    dep_task = []
    dep_areas = []
    path2_all_copy = []  # 存储每个子区域四个方向上的航道
    #  将每一个区域的路程分割成子区域，每个子区域的路程小于等于最小路程
    m = 10  # 颜色
    for i in range(0, area_num):
        point = path_case1[i]  # print(np.array(point))
        dis0 = distance_all1[i]  # print(dis0)
        # 1.剩余里程非常少，可以不分解
        if s_all1[i]-min_s <= min_s/5:
            # print("区域保留")
            sub_path = []
            sub_path.append(point)
            sub_s = [s_all1[i]]
        else:
            sub_path, sub_s, d_all = get_sub_area(point, dis0, min_s)  # print(np.array(sub_path), len(sub_path))
            if sub_s[-1] < min_s/5:
                # print('区域合并')
                sub_path = com_path(sub_path)
                d_all = com_path(d_all)
                # 最后两个点求和
                sum_s = np.array(sub_s[-2])+np.array(sub_s[-1])
                sub_s.pop(-1)
                sub_s.pop(-1)
                sub_s.append(sum_s)
        for j in range(len(sub_path)):
            # 获得所有方向的航道
            sub_dep_task = []
            node_all = get_all_path(sub_path[j])
            path2_all_copy.append(node_all)
            for jj in range(4):
                sub_dep_task.append([node_all[jj][0], node_all[jj][-1]])
            # 绘图
            x = np.array(sub_path[j])[:, 0]
            y = np.array(sub_path[j])[:, 1]
            np_sub_path = np.array(sub_path[j])
            start_yaw = plot_turn(np_sub_path, curvature, color[m], ax3)
            dep_task.append(np_sub_path)  # 存储的是航路点
            s_task.append([sub_s[j]])  # 存储的是路程长度
            if j == 0:
                ax3.plot(x[0], y[0], marker='*', color=color[m], markersize=10, label=str(np.round(sub_s[j])))
                ax3.plot(x[-1], y[-1], marker='o', color=color[m], markersize=6)
            elif j == len(sub_path)-1:
                ax3.plot(x[0], y[0], marker='o', color=color[m], markersize=6)
                ax3.plot(x[-1], y[-1], marker='x', color=color[m], markersize=10, label=str(np.round(sub_s[j])))  # , label='end'
            else:
                ax3.plot(x[-1], y[-1], marker='o', color=color[m], markersize=6, label=str(np.round(sub_s[j])))
            m = m + 1
            start_position = [x[-1], y[-1], start_yaw]
            dep_areas.append(sub_dep_task)
    ax3.title.set_text("Task Decomposition")
    ax3.legend()
    ax3.set_aspect(1)    #
    ax3.grid()
    # print("存储的是航路点", np.array(dep_task))
    # print("存储的是路程长度", s_task)
    return dep_areas, s_task, path2_all_copy


def GA_vrp(map,ax4,order,start,path_all,curvature):
    map.map_plot(ax4)
    ax4.plot(start[0], start[1], 'ks', markersize=10)  # 无人机的起点, label='initial'
    co = ['-y', '-m', '-k']
    co_c = ['y', 'm', 'k']
    '''获得多条航路'''
    order0 = [int(i) for i in order]
    order1 = [int(i*10 % 10) for i in order]
    # 将访问顺序转化为多条路径
    order_i = []
    order_j = []
    index_1 = num_e(order0, -1)  # order0中-1的索引
    for i, j in zip(index_1[0::], index_1[1::]):
        o1 = order0[i + 1:j]
        o2 = order1[i + 1:j]
        order_i.append(o1)
        order_j.append(o2)


    distance_i = []
    i ,m= 0,10
    # 顺序调换带来区域搜索航程的改变可以忽略
    for index_i, path_index_i in zip(order_i, order_j):
        start_position = start.copy()
        path_length_i = 0
        for index, path_index in zip(index_i, path_index_i):
            # 根据当前位置选择一个最近的航道,绘制到达区域的路径
            node_pos = path_all[index][path_index]
            # 计算距离
            path_length = plot_initial(start_position, node_pos, curvature, ax4, co[i])
            distance, s = get_distance(node_pos, curvature)
            path_length_i = path_length_i + path_length + s
            # 航路点的x和y,搜索完后，无人机的位置需要更新
            x = np.array(node_pos)[:, 0]
            y = np.array(node_pos)[:, 1]
            start_yaw = plot_turn(node_pos, curvature, color[m], ax4)
            start_position = [x[-1], y[-1], start_yaw]
            m = m + 1
        ax4.plot(start[0], start[1], markersize=10, color=co_c[i],label='UAV' + str(i) + ":" + str(path_length_i))  # 无人机的起点, label='initial'
        i = i + 1
        distance_i.append(path_length_i)
        path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(start_position[0],start_position[1],start_position[2],
                                                                           start[0], start[1], pi, curvature)
    # 绘图
    ax4.title.set_text("Genetic Algorithm——" + 'Total distance=' + str(distance_i))
    ax4.set_aspect(1)  #
    ax4.grid()
    ax4.legend()
    return order_i, order_j, distance_i

'''1.区域覆盖'''
def get_main():
    start, lx, curvature = get_config()
    # 生成地图，获得各个区域的信息点
    map = generate_map(2)
    area_all = map.area_all
    '''获得各个区域的覆盖航路，起始点/离开点，进入角/离开角'''
    path_all,area_task,yaw_task = get_cover_path(area_all,lx)
    # print("起始点/离开点")
    # print(area_task)
    # 获得各个城市的位置信息,用于强化学习测试
    area_task1 = np.array(area_task)
    city_task = np.zeros((4,2))
    for i in range(np.shape(area_task1)[0]):
        area_task2 = area_task1[i,0:2,:,:].reshape(4,2)
        city_task = np.vstack((city_task,area_task2))
    city_task = city_task[4:] #(24, 2)
    return city_task,path_all,area_task,yaw_task

# '''def main():
#     start, lx, curvature = get_config()
#     map = generate_map(2)
#     area_all = map.area_all
#     area_num = len(area_all)
#     # 获取参数配置,start包含起始位置和角度
#     co = ['-y', '-m', '-k']
#     co_c = ['y', 'm', 'k']

#     fig, axes = plt.subplots(1, 3, figsize=(10, 8))   # 定义子区间的个数，注意第一个fig后面是逗号
#     ax1 = axes[0]  # 第一个图位置
#     ax2 = axes[1]  # 第二个图位置
#     ax3 = axes[2]  # 第三个图位置
#     plt.xlim(-20, 120)
#     plt.ylim(-20, 100)

#     '''1.区域覆盖'''
#     '''获得各个区域的覆盖航路，起始点/离开点，进入角/离开角'''
#     city_task, path_all, area_task, yaw_task = get_main()

#     '''2.单机的区域覆盖顺序'''

#     '''2.1 贪婪算法获得区域覆盖顺序，返回的是：存储各个区域的总航程、航路点、各个区域的各个距离信息（直线，转弯）'''
#     s_all1, path_case1, distance_all1 = greedy(map, ax1, area_num, start, path_all, curvature)
#     # plt.show()
#     '''2.2 遗传算法获得区域覆盖顺序，返回的是：访问各个区域的顺序，以及总航程（包含返回原点的距离）'''
#     order, dis_length2, lx,ly = ga_order(area_task, yaw_task, start[0:2], curvature)
#     ga_tsp(map, ax2, start, path_all, curvature, order)
#     # ax2.scatter(lx, ly, s=80, c='k')
#     # ax2.plot(lx, ly, 'ro-')
#     ax2.title.set_text("Genetic Algorithm——"+'Total distance='+str(np.round(dis_length2)))
#     plt.show()

#     '''2.2 学习算法获得区域覆盖顺序，返回的是：访问各个区域的顺序，以及总航程（包含返回原点的距离）'''
#     order, dis_length2 = ga_order(area_task, yaw_task, start[0:2], curvature)
#     ga_tsp(map, ax3, start, path_all, curvature, order)
#     ax3.title.set_text("Genetic Algorithm——"+'Total distance='+str(np.round(dis_length2)))

#     plt.show()
#     return area_task, yaw_task, start, curvature


# if __name__ == '__main__':
#     main()




