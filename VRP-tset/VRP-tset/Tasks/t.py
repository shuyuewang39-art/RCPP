import numpy as np
import matplotlib.path as mpath
from Tasks.dubins import dubins_path_planning
pi = 3.1415926


def choose_path(path_all, position, curvature):
    # 根据无人机当前的位置选择一个最近的航道
    length_all = []
    for j in range(len(path_all)):
        node_all = path_all[j]
        for i in range(4):
            node_pos = node_all[i]
            end_yaw = np.arctan((node_pos[1][1] - node_pos[0][1]) / (node_pos[1][0] - node_pos[0][0])) if node_pos[0][0] != node_pos[1][0] else pi/2# -pi # -pi
            if node_pos[1][0] < node_pos[0][0]:  # 在二三象限
                end_yaw = end_yaw + pi
            path_x, path_y, path_yaw, mode, path_length = dubins_path_planning(position[0], position[1],
                                                                           position[2], node_pos[0][0],
                                                                           node_pos[0][1], end_yaw, curvature)
            length_all.append(path_length)
    index_min = np.argmin(length_all)

    area_index = int(index_min / 4) # 获得最近的区域
    path_index = index_min % 4 # 获得最近的方向
    path_num = area_index+path_index/10  # 整数部分表示区域，小数部分表示方向，用于遗传编码
    node_pos = path_all[area_index][path_index] # 选择的区域的航迹点
    node_temp = (node_pos.copy()).tolist()
    # 将选择的从列表中删去
    path_all.pop(area_index)
    return node_temp, path_num


def get_sub_area(point_np, dis0, min_distance):
    s = 0
    path_temp = []
    d_temp = []
    path_all = []
    d_all = []
    dis = dis0.copy()
    point = point_np.copy()  # point = point.tolist()
    s_ = min_distance
    s_all = []
    while dis:
        if len(dis) == 1:
            dis.insert(1, 0)
            # print("------------1-------------")
        if len(dis) == 0:
            break
        d = dis[0]+dis[1]  # d = np.linalg.norm(np.array(point[0])-np.array(point[1]))  # 当前航道的路程
        s = s+d
        if (s >= s_ or dis[-1] == 0) and s-d < s_:
            delta_d = s_-(s-d)  # 差值
            # 最后一条航道
            if dis[-1] == 0:
                path_temp.append(point[0])
                path_temp.append(point[1])

                point.pop(0)
                point.pop(0)

                d_temp.append(dis[0])
                dis.pop(0)
                dis.pop(0)

            elif delta_d < dis[0] / 5:  # 断点在直线中点左边
                s = s-turn_l
                d_temp.pop(-1)

            # elif delta_d < 9*dis[0]/10:  # 断点在直线处，但是不能靠近拐弯处
                # if delta_d < dis[0]/10:  # 只占据下一条航道非常小的一段距离
                #     s = s-turn_l
                #     d_temp.pop(-1)
                # else:
                #     thet = np.arccos((point[1][0] - point[0][0]) / dis[0])
                #     theta = thet if point[1][1] - point[0][1] > 0 else -thet
                #     end_point = point[0] + delta_d*np.array([np.cos(theta), np.sin(theta)])
                #     left_d = dis[0] - delta_d
                #
                #     path_temp.append(point[0])
                #     path_temp.append(end_point)
                #
                #     point.pop(0)
                #     point.insert(0, end_point)
                #     s = s - turn_l-left_d
                #     # d_temp.append(dis[0])
                #     d_temp.append(delta_d)
                #
                #     dis.pop(0)
                #     dis.insert(0, left_d)

            else:  # 断点在拐弯处
                end_point = point[1]
                s = s - turn_l
                d_temp.append(dis[0])
                dis.pop(0)
                dis.pop(0)

                path_temp.append(point[0])
                path_temp.append(end_point)
                point.pop(0)
                point.pop(0)
            # print("------2----------")
            path_all.append(path_temp)
            d_all.append(d_temp)
            s_all.append(s)
            s = 0
            path_temp = []
            d_temp = []
        else:
            path_temp.append(point[0])
            path_temp.append(point[1])
            point.pop(0)
            point.pop(0)

            d_temp.append(dis[0])
            d_temp.append(dis[1])
            dis.pop(0)
            turn_l = dis[0]
            dis.pop(0)
    return path_all, s_all, d_all


def ro_p(ro_convhull):
    Path = mpath.Path
    path_data = []
    for i in range(len(ro_convhull)):
        if i == 0:
            path_data.append((Path.MOVETO, ro_convhull[i]))
        else:
            path_data.append((Path.LINETO, ro_convhull[i]))
    path_data.append((Path.LINETO, ro_convhull[0]))
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    return path


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


def get_distance(node_pos, curvature):
    #  杜宾斯拐弯曲线，起点序号为1，3，5,...；对应的终点序号为2,4,6...
    distance = []
    theta = np.arctan((node_pos[1][1]-node_pos[0][1])/(node_pos[1][0]-node_pos[0][0])) if node_pos[0][0] != node_pos[1][0] else -pi/2# -pi
    if node_pos[1][0] < node_pos[0][0]:  # 在二三象限
        theta = theta+pi
    start_yaw = theta
    end_yaw = start_yaw + pi
    for i in range(np.shape(node_pos)[0]):
        # 输入分别是起点和终点的位置，角度
        if i*2+2 >= np.shape(node_pos)[0]:
            break
        path_x, path_y, path_yaw, mode, path_length = dubins_path_planning(node_pos[i*2+1][0], node_pos[i*2+1][1],
                                                                           start_yaw, node_pos[i*2+2][0],
                                                                           node_pos[i*2+2][1], end_yaw, curvature)
        ar = start_yaw
        start_yaw = end_yaw
        end_yaw = ar
        # 存储直线和转弯处的长度
        d = np.linalg.norm(np.array(node_pos[i*2]) - np.array(node_pos[i*2+1]))  # 直线距离
        distance.append(d)
        distance.append(path_length)
    # 绘制最后一条不带拐弯的直线
    d = np.linalg.norm(np.array(node_pos[i * 2]) - np.array(node_pos[i * 2 + 1]))  # 直线距离
    distance.append(d)
    s = sum(distance)
    return distance, s

