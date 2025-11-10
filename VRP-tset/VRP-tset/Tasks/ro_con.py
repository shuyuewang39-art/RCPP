import math
import numpy as np
import matplotlib.path as mpath
M_PI = math.pi


def StripPathGenerator(convhull_vertexes, lx):
    # 获得凸包在x轴上的范围
    min_x = convhull_vertexes[0][0]
    max_x = convhull_vertexes[0][0]
    for i in range(len(convhull_vertexes)):
        if min_x > convhull_vertexes[i][0]:
            min_x = convhull_vertexes[i][0]  # 凸包中最小的x
        if max_x < convhull_vertexes[i][0]:
            max_x = convhull_vertexes[i][0]  # 凸包中最大的x
    areaWidth = max_x - min_x  # 凸包的最大宽度
    # 找到旋转矩阵的旋转角度
    thetaMin = 0
    for i in range(0, 360, 1):
        # theta = i*2.0*M_PI/360.0  # 角度
        theta = i * 2.0 * M_PI / 360.0  # 角度
        R11 = np.cos(theta)
        R12 = -np.sin(theta)
        max_aux_x = R11*convhull_vertexes[0][0] + R12*convhull_vertexes[0][1]
        min_aux_x = R11*convhull_vertexes[0][0] + R12*convhull_vertexes[0][1]
        # 所有的边界顶点向某个方向投影，找出最大值和最小值
        for j in range(len(convhull_vertexes)):
            aux_x = R11 * convhull_vertexes[j][0] + R12 * convhull_vertexes[j][1]
            if min_aux_x > aux_x:
                min_aux_x = aux_x
            if max_aux_x < aux_x:
                max_aux_x = aux_x
        # 根据投影中的最大差值重新确定凸包的最大宽度和边界，并且找出这个方向向量
        if max_aux_x - min_aux_x < areaWidth:
            areaWidth = max_aux_x - min_aux_x
            min_x = min_aux_x
            max_x = max_aux_x
            thetaMin = theta
    # print("thetaMin:"+str(thetaMin))
    # 获得旋转矩阵
    R11 = np.cos(thetaMin)
    R12 = -np.sin(thetaMin)
    R21 = np.sin(thetaMin)
    R22 = np.cos(thetaMin)
    # 确定凸包的高度范围
    min_y = R21 * convhull_vertexes[0][0] + R22 * convhull_vertexes[0][1]
    max_y = R21 * convhull_vertexes[0][0] + R22 * convhull_vertexes[0][1]
    for i in range(len(convhull_vertexes)):
        aux_y = R21 * convhull_vertexes[i][0] + R22 * convhull_vertexes[i][1]
        if min_y > aux_y:
            min_y = aux_y
        if max_y < aux_y:
            max_y = aux_y
    # 根据上述信息将这个凸包进行了旋转,获得旋转后的凸包
    rotate_convhull_vertexes = convhull_vertexes.copy()
    for i in range(len(convhull_vertexes)):
        rotate_convhull_vertexes[i][0] = R11 * convhull_vertexes[i][0] + R12 * convhull_vertexes[i][1]
        rotate_convhull_vertexes[i][1] = R21 * convhull_vertexes[i][0] + R22 * convhull_vertexes[i][1]
    #  1111
    # R11_inv = np.cos(thetaMin)
    # R12_inv = np.sin(thetaMin)
    # R21_inv = -np.sin(thetaMin)
    # R22_inv = np.cos(thetaMin)
    # # 根据上述信息将这个凸包进行了旋转,获得旋转后的凸包
    # rotate_convhull = convhull_vertexes.copy()
    # for i in range(len(convhull_vertexes)):
    #     rotate_convhull[i][0] = R11_inv * rotate_convhull_vertexes[i][0] + R12_inv * rotate_convhull_vertexes[i][1]
    #     rotate_convhull[i][1] = R21_inv * rotate_convhull_vertexes[i][0] + R22_inv * rotate_convhull_vertexes[i][1]
    #     # print("------------------------------")
    #     # print(convhull_vertexes[i])
    #     # print(rotate_convhull[i])
    #     # print("------------------------------")
    path, node_pos, path_r = path_get(areaWidth, thetaMin, min_x, min_y, max_y, rotate_convhull_vertexes, lx)
    return rotate_convhull_vertexes, path, node_pos, path_r

def path_get(areaWidth,thetaMin,min_x,min_y,max_y,rotate_convhull_vertexes,lx):
    # 几个航道
    m_imageWidth = lx
    m_imageLength = 0.1
    numLanes = np.ceil(areaWidth / m_imageWidth)
    delta = m_imageLength
    # 每个航道的宽度
    laneDist = areaWidth / numLanes
    R11_inv = np.cos(thetaMin)
    R12_inv = np.sin(thetaMin)
    R21_inv = -np.sin(thetaMin)
    R22_inv = np.cos(thetaMin)
    # print(R11_inv,R12_inv,R21_inv,R22_inv)
    path_data = []
    path_data_r = []
    # 存储关键结点
    node_pos = []
    for i in range(int(numLanes)):# int(numLanes)
        # 航道是左右对称的，这里计算的其中心轴线位置
        xi = min_x + laneDist*(i+1) - laneDist/2.0
        # 这里首先假定以最大的深度搜索
        min_yi = min_y
        max_yi = max_y
        count = (max_y - min_y)/delta
        # 判断该深度是否在凸包内，如果不再凸包内，就在起始深度的基础上增加一段距离
        while isPoiWithinPoly([xi, min_yi], rotate_convhull_vertexes) == False and count > 0:
            min_yi += delta
            count = count-1
        if count <= 0:
            continue  # 终止最近的循环中的当前迭代并立即开始下一次迭代
        count = (max_y - min_y)/delta
        # 判断该深度是否在凸包内，如果不再凸包内，就在终止深度的基础上减小一段距离
        while isPoiWithinPoly([xi, max_yi], rotate_convhull_vertexes) == False and count > 0:
            max_yi -= delta
            count = count-1
        if count <= 0:
            continue
        # 轨道是一条直线，xi表示其在x轴的位置，min_yi和max_yi表示这条线有多长
        # 起始点和终止点的位置坐标
        lane_p1_x = R11_inv*xi + R12_inv*min_yi
        lane_p1_y = R21_inv*xi + R22_inv*min_yi
        lane_p2_x = R11_inv*xi + R12_inv*max_yi
        lane_p2_y = R21_inv*xi + R22_inv*max_yi
        # print(xi, min_yi)
        # print(lane_p1_x, lane_p1_y)
        Path = mpath.Path
        if i == 0:
            path_data.append((Path.MOVETO, (lane_p1_x, lane_p1_y)))
            path_data.append((Path.LINETO, (lane_p2_x, lane_p2_y)))
            path_data_r.append((Path.MOVETO, (xi, min_yi)))
            path_data_r.append((Path.LINETO, (xi, max_yi)))
            node_pos.append([lane_p1_x, lane_p1_y])
            node_pos.append([lane_p2_x, lane_p2_y])
            # dubins_path_planning(start_x, start_y, start_yaw,
            #                      end_x, end_y, end_yaw, curvature)
        elif i % 2 == 1:
            path_data.append((Path.MOVETO, (lane_p2_x, lane_p2_y)))
            path_data.append((Path.MOVETO, (lane_p1_x, lane_p1_y)))
            path_data_r.append((Path.MOVETO, (xi, max_yi)))
            path_data_r.append((Path.MOVETO, (xi, min_yi)))
            node_pos.append([lane_p2_x, lane_p2_y])
            node_pos.append([lane_p1_x, lane_p1_y])
        else:
            path_data.append((Path.MOVETO, (lane_p1_x, lane_p1_y)))
            path_data.append((Path.LINETO, (lane_p2_x, lane_p2_y)))
            path_data_r.append((Path.MOVETO, (xi, min_yi)))
            path_data_r.append((Path.MOVETO, (xi, max_yi)))
            node_pos.append([lane_p1_x, lane_p1_y])
            node_pos.append([lane_p2_x, lane_p2_y])
    codes, verts = zip(*path_data)
    codesr, vertsr = zip(*path_data_r)
    path = mpath.Path(verts, codes)
    path_r = mpath.Path(vertsr, codesr)
    return path, node_pos, path_r


def isRayIntersectsSegment(poi, s_poi, e_poi): #[x,y] [lng,lat]
    #输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    # 边与射线重合s_poi[1] == e_poi[1] and poi[1] == e_poi[1]
    # 边与射线平行 s_poi[1] == e_poi[1]
    if s_poi[1] == e_poi[1] and poi[1] == e_poi[1]:  # 排除与射线重合，线段首尾端点重合的情况
        return False
    if s_poi[1] == e_poi[1] and poi[1] != e_poi[1]:  # 排除与射线平行，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]: #线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]: #线段在射线下边
        return False
    if s_poi[0] < poi[0] and e_poi[0] < poi[0]: #线段在射线左边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]: #交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]: #交点为下端点，对应epoint
        return False
    xseg = e_poi[0]-(e_poi[0]-s_poi[0])*(e_poi[1]-poi[1])/(e_poi[1]-s_poi[1]) #求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  #排除上述情况之后

# 修改后
def pointInLine(point1,point2,pointQ):
    x1, y1 = point1
    x2, y2 = point2
    xQ, yQ = pointQ
    maxX = max(x1, x2)
    maxY = max(y1, y2)
    minX = min(x1, x2)
    minY = min(y1, y2)
    if (abs(((xQ-x1)*(y2-y1) - (x2-x1)*(yQ-y1))) < 0.00001) and (xQ >= minX and xQ <= maxX) and (yQ >=minY and yQ <= maxY):
        return True
    else:
        return False

def isPoiWithinPoly(poi,poly):
    #输入：点，多边形三维数组
    #poly=[[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]]] 三维数组
    #可以先判断点是否在外包矩形内
    #if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    #但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc=0 #交点个数
    # for epoly in poly: #循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    # 首先判断点是否在边框上
    flag1 = False
    for i in range(len(poly)): #[0,len-1]
        s_poi = poly[i, :]
        if i+1 == len(poly):
            e_poi = poly[0, :]
        else:
            e_poi = poly[i+1, :]
        if pointInLine(s_poi, e_poi, poi):
            flag1 = True
            break
    # 检查是否在边框内部
    for i in range(len(poly)): #[0,len-1]
        s_poi = poly[i, :]
        if i+1 == len(poly):
            e_poi = poly[0, :]
        else:
            e_poi = poly[i+1, :]
        if isRayIntersectsSegment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1
    flag2 = True if sinsc % 2 == 1 else False
    flag = flag1 or flag2
    # print(flag1, flag2)
    return flag
