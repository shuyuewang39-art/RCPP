import numpy as np
from Tasks.dubins import dubins_path_planning
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

pi = 3.1415926

#  画多边形时，需要顺着（逆）时针填写多边形的角坐标
# area1 = [[10, 0], [10, 35], [40, 20]]
# area2 = [[60, 30], [80, 30], [90, 10], [70, 10]]
# area3 = [[12, 50], [5, 65], [20, 80], [35, 65], [28, 50]]
area1 = [[0.0, 0.0], [0.0, 35], [20, 45], [30, 15]]  #
area2 = [[60, 30], [80, 30], [90, 10], [70, 10]]
area3 = [[12, 50], [5, 65], [20, 80], [35, 65], [28, 50]]
area4 = [[80, 40], [100, 40], [100, 60], [80, 60]]
# 案例2：
# area1 = [[0.0, 0.0], [0.0, 25], [20, 25], [20, 0]]  #
# area2 = [[60, 30], [80, 30], [80, 10], [70, 10]]
# area3 = [[12, 50], [5, 65], [20, 80], [35, 75], [28, 50]]
# area4 = [[80, 40], [100, 40], [115, 75], [80, 60]]
# 案例3
# area1 = [[0.0, 0.0], [0.0, 25], [20, 15], [20, 0]]  #
# area2 = [[60, 30], [80, 20], [100, 10], [60, 0]]
# area3 = [[12, 60], [5, 65], [20, 80], [35, 75], [28, 50]]
# area4 = [[90, 40], [90, 60], [110, 60], [110, 40]]
class generate_map(object):
    def __init__(self, area_num):
        self.area_all = []
        self.area_num = 2

        area1_np = np.array(area1)
        area2_np = np.array(area2)
        area3_np = np.array(area3)
        area4_np = np.array(area4)

        float_arr1 = area1_np.astype(np.float64)
        float_arr2 = area2_np.astype(np.float64)
        float_arr3 = area3_np.astype(np.float64)
        float_arr4 = area4_np.astype(np.float64)

        self.area_all.append(float_arr1)
        self.area_all.append(float_arr2)
        self.area_all.append(float_arr3)
        self.area_all.append(float_arr4)

        self.hull_t = []
        self.point = []
        for i in range(area_num):
            np.random.seed(i+2) #2+i+5
            a = [[np.random.randint(30+i*20, 60+i*20) for j in range(2)] for k in range(12)]  # 一个12行2列的数组
            points = np.array(a, dtype='float64')
            self.point.append(points)
            hull = ConvexHull(points)  # hull.vertices存储的是凸包顶点的坐标
            self.hull_t.append(hull)
            convhull_point = points[hull.vertices, :]  # hull.simplices存储的是凸包边的坐标
            self.area_all.append(convhull_point)

    def map_plot(self, axes):
        c=['y','slateblue']
        # 地图边界,长方形区域
        square = plt.Rectangle(xy=(-10, -10), width=130, height=100, alpha=0.2, angle=0.0, linestyle='-', linewidth=2,
                               edgecolor='k')  # xy: 左下角位置，width, height：长，宽，angle：逆时针旋转角度,facecolor='none'
        axes.add_patch(square)  # 把图形加载到绘制区域
        p3 = plt.Polygon(xy=area1, edgecolor='k',  alpha=0.2, facecolor='r')
        p4 = plt.Polygon(xy=area2, alpha=0.2, edgecolor='k', facecolor='g')
        p5 = plt.Polygon(xy=area3, alpha=0.2, edgecolor='k', facecolor='b')
        p6 = plt.Polygon(xy=area4, alpha=0.2, edgecolor='k', facecolor='y')
        axes.add_patch(p3)
        axes.add_patch(p4)
        axes.add_patch(p5)
        axes.add_patch(p6)
        for i in range(len(self.hull_t)):
            convhull_point = self.area_all[i+4]
            x = convhull_point[:,0].tolist()
            y = convhull_point[:,1].tolist()
            axes.fill(x, y, color=c[i], alpha=0.2)
            for simplex in self.hull_t[i].simplices:
                axes.plot(self.point[i][simplex, 0], self.point[i][simplex, 1], 'k-', alpha=0.2)  # 绘制边框


def plot_initial(start_position,node_pos,curvature,ax,colo):
    # 无人机到达航迹起点的轨迹
    theta = np.arctan((node_pos[1][1]-node_pos[0][1])/(node_pos[1][0]-node_pos[0][0])) if node_pos[0][0] != node_pos[1][0] else -pi/2# -pi
    if node_pos[1][0] < node_pos[0][0]:  # 在二三象限
        theta = theta+pi
    if theta == -pi/2 and node_pos[1][1] > node_pos[0][1]:
        theta = theta + pi
    end_yaw = theta
    path_x0, path_y0, path_yaw0, mode0, path_length0 = dubins_path_planning(start_position[0], start_position[1],
                                                                       start_position[2], node_pos[0][0],
                                                                       node_pos[0][1], end_yaw, curvature)
    # ax.plot(start_position[0], start_position[1], 'ks', markersize=10)  # 无人机的起点, label='initial'
    # print(start_position, node_pos[0], end_yaw, path_length0)
    path_length = np.linalg.norm(np.array(start_position[0:2]) - np.array(node_pos[0]))
    ax.plot(path_x0, path_y0, 'r-')  # 到达任务区域的航迹点, 'm--'
    # ax.scatter(path_x0[0], path_x0[0],  c='r')
    # ax.scatter(path_x0[-1], path_x0[-1],  c='r')
    # ax.plot(lx, ly, )colo
    return path_length


# 绘制箭头
def plot_arrow(x, y, yaw, ax, length=1.5, width=1, fc="k", ec="k", ):
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        ax.arrow(x, y, length * np.math.cos(yaw), length * np.math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        ax.plot(x, y)


def plot_turn(node_pos, curvature, str, ax):
    path_turn = []
    theta = np.arctan((node_pos[1][1]-node_pos[0][1])/(node_pos[1][0]-node_pos[0][0])) if node_pos[0][0] != node_pos[1][0] else -pi/2# -pi
    if node_pos[1][0] < node_pos[0][0]:  # 在二三象限
        theta = theta+pi
    if theta == -pi/2 and node_pos[1][1] > node_pos[0][1]:
        theta = theta + pi
    start_yaw = theta
    end_yaw = start_yaw + pi
    for i in range(np.shape(node_pos)[0]):
        # 输入分别是起点和终点的位置，角度
        if i*2+2 >= np.shape(node_pos)[0]:
            break
        start_x = node_pos[i*2+1][0]
        start_y = node_pos[i * 2 + 1][1]
        end_x = node_pos[i*2+2][0]
        end_y = node_pos[i * 2 + 2][1]
        path_x, path_y, path_yaw, mode, path_length = dubins_path_planning(node_pos[i*2+1][0], node_pos[i*2+1][1],
                                                                           start_yaw, node_pos[i*2+2][0],
                                                                node_pos[i*2+2][1], end_yaw, curvature)
        # 直线轨迹
        ax.plot(node_pos[i*2:i*2+2, 0], node_pos[i*2:i*2+2, 1], str)
        # 转弯轨迹
        ax.plot(path_x, path_y, str)
        # 箭头
        plot_arrow(start_x, start_y, start_yaw, ax)
        plot_arrow(end_x, end_y, end_yaw, ax)
        # 方向的转换
        ar = start_yaw
        start_yaw = end_yaw
        end_yaw = ar
        # 存储转弯处的轨迹点
        path_turn.append([path_x, path_y])
    # 绘制最后一条不带拐弯的直线
    # if i == 0:
    #     print(node_pos)
    #     print(node_pos[i * 2:i * 2 + 2, 0])
    ax.plot(node_pos[i * 2:i * 2 + 2, 0], node_pos[i * 2:i * 2 + 2, 1], linestyle='-', color=str)
    # plt.scatter(node_pos[:, 0], node_pos[:, 1], color='r')
    return start_yaw