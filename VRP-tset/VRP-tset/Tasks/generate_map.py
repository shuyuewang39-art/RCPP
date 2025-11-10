from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

# 在一个长方形区域生成多个需要搜索的区域
def generrate(area_num):
    # 地图边界,长方形区域
    square = plt.Rectangle(xy=(-10, 0), width=120, height=100, alpha=0.2, angle=0.0, linestyle='-', linewidth=2,
                           edgecolor='k')  # xy: 左下角位置，width, height：长，宽，angle：逆时针旋转角度,facecolor='none'
    area_all = []
    area_num = 2
    fig, axes = plt.subplots()
    axes.add_patch(square)  # 把图形加载到绘制区域
    #  画多边形时，需要顺着（逆）时针填写多边形的角坐标
    # area1 = [[0.0, 0.0], [0.0, 35], [20, 45], [30, 15]]  #
    # area2 = [[60, 30], [80, 30], [90, 10], [70, 10]]
    # area3 = [[12, 50], [5, 65], [20, 80], [35, 65], [28, 50]]
    # area4 = [[80, 40], [100, 40], [100, 60], [80, 60]]
    # 案例2：
    area1 = [[0.0, 0.0], [0.0, 25], [20, 25], [20, 0]]  #
    area2 = [[60, 30], [80, 30], [80, 10], [70, 10]]
    area3 = [[12, 50], [5, 65], [20, 80], [35, 75], [28, 50]]
    area4 = [[80, 40], [100, 40], [115, 75], [80, 60]]
    area_all.append(np.array(area1))
    area_all.append(np.array(area2))
    area_all.append(np.array(area3))
    area_all.append(np.array(area4))

    p3 = plt.Polygon(xy=area1, alpha=0.2, edgecolor='k', facecolor='r')
    p4 = plt.Polygon(xy=area2, alpha=0.2, edgecolor='k', facecolor='g')
    p5 = plt.Polygon(xy=area3, alpha=0.2, edgecolor='k', facecolor='b')
    p6 = plt.Polygon(xy=area4, alpha=0.2, edgecolor='k', facecolor='y')

    axes.add_patch(p3)
    axes.add_patch(p4)
    axes.add_patch(p5)
    axes.add_patch(p6)

    for i in range(area_num):
        np.random.seed(i+5)  # i+2
        a = [[np.random.randint(30+i*20, 60+i*20) for j in range(2)] for k in range(12)]  # 一个12行2列的数组
        points = np.array(a)
        hull = ConvexHull(points)  # hull.vertices存储的是凸包顶点的坐标
        convhull_point = points[hull.vertices, :]  # hull.simplices存储的是凸包边的坐标
        area_all.append(convhull_point)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')  # 绘制边框
        plt.plot(points[:, 0], points[:, 1], 'o')  # 绘制原始点
    plt.xlim(-10, 110)
    plt.ylim(-10, 90)
    # plt.show()
    return area_all

if __name__ == '__main__':
    generrate(2)
