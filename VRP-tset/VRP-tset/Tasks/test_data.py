import numpy as np
from Tasks.config import get_config,generate_map
from Tasks.ro_con import StripPathGenerator
pi = 3.1415926

def get_all_path(node_pos):
    node_pos0 = np.array(node_pos)
    node_pos2 = node_pos0[::-1]
    node_pos1 = node_pos0.copy()
    node_pos3 = node_pos0.copy()
    for i in range(int(np.shape(node_pos)[0]/2)):
        node_pos1[i * 2+1], node_pos1[i * 2] = node_pos0[i * 2], node_pos0[i * 2 + 1]
        node_pos3[i * 2 + 1], node_pos3[i * 2] = node_pos2[i * 2], node_pos2[i * 2 + 1]
    node_all = [node_pos0, node_pos1, node_pos2, node_pos3]
    return node_all

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

'''1.区域覆盖'''
def get_main():
    start, lx, curvature = get_config()
    # 生成地图，获得各个区域的信息点
    map = generate_map(2)
    area_all = map.area_all
    '''获得各个区域的覆盖航路，起始点/离开点，进入角/离开角'''
    path_all,area_task,yaw_task = get_cover_path(area_all,lx)
    # 获得各个城市的位置信息,用于强化学习测试
    area_task1 = np.array(area_task)
    city_task = np.zeros((4,2))
    for i in range(np.shape(area_task1)[0]):
        area_task2 = area_task1[i,0:2,:,:].reshape(4,2)
        city_task = np.vstack((city_task,area_task2))
    city_task = city_task[4:] #(24, 2)
    return city_task,path_all,area_task,yaw_task




