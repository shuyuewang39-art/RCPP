from copy import deepcopy
import numpy as np
import random
import math
import torch
import matplotlib.pyplot as plt
from init import global_value


def reward(static, tour_indices):
    static = torch.Tensor(static)
    idx = tour_indices.unsqueeze(1).expand(-1, int(static.size(1)/2), -1)   #将tour_indices复制了 torch.Size([256, 4, 13])
    # tour按照这一回合选择的城市的索引获得其二维坐标按顺序排列的的结果
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # torch.Size([256, step, 4])  
    # start表示每个样本的起始位置的坐标torch.Size([256, 1，8])
    start = static.data[:, 0:2, 0] # torch.Size([256, 2])
    start = torch.cat([start,start],dim=1).unsqueeze(1) #torch.Size([256, 1, 4])
    # y表示一个完整来回的位置序列表示torch.Size([256, 1+step+1，4])
    y = torch.cat((start, tour, start), dim=1) #按维数1（列）拼接，行不变列变（0，1，...5,0）
    # y[:, :-1]取原始数据的除最后一个元素之外的值（0，1，...5）
    # y[:, 1:]取原始数据的除第一个元素之外的值（1，...5，0）
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1, 2:4] - y[:, 1:,0:2], 2), dim=2))
    return tour_len.sum(1)


# 因为area只存储了位置不包含远点，因此第一个区域的下表应该是0
def calmile(g0,f=None,g1=None):

    static = dataDict['distance']
    area_num = int((np.shape(static)[1]-1)/4)
    area = get_ga_in(area_num, static)
    start_city = static[0:2,0]
    start = start_city[0:2]

    city = [0, 1]
    city_ = [0, 1]

    # 当前城市及其起点和终点
    city[0] = int(math.modf(g0)[1])-1 # 区域索引
    city[1] = int(np.round(math.modf(g0)[0] * 10)) # 方向索引
    point0 = area[city[0]][city[1]][0]
    point1 = area[city[0]][city[1]][1]
    dis0 = np.linalg.norm(point0 - start)
    dis2 = np.linalg.norm(point1 - start)
    if f==0:
        return dis0,dis2
    elif f==1:
        # 目标城市及其起点和终点
        city_[1] = int(np.round(math.modf(g1)[0] * 10))
        city_[0] = int(math.modf(g1)[1])-1
        point0_ = area[city_[0]][city_[1]][0]
        point1_ = area[city_[0]][city_[1]][1] 
        dis1 = np.linalg.norm(point0_-point1)
        dis2 = np.linalg.norm(point1_ - start)
        return dis0,dis1,dis2
    elif f==2:
        dis0 = np.linalg.norm(point1 - start)
        return dis0


def get_ga_in(area_num,static):
    # static:torch.Size([batch, 8, area_num*4+1])-torch.Size([8, area_num*4+1])
    area_task = [[] for _ in range(area_num)]
    for i in range(0, area_num): 
        for j in range(4):
            area_task[i].append([static[0:2,i*4+j+1], static[2:4,i*4+j+1]])
    return area_task

dataDict = global_value.dataDict
max_num = dataDict['des_pos']


'''生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的'''
# 生成满足运载量的任务序列
def genInd():
    # 城市数量
    dataDict = global_value.dataDict
    nCustomer = np.shape(dataDict['NodeCoor'])[0] - 1  
    # 生成城市的随机排列,注意城市编号为1--n，覆盖方向编号为0-1-2-3
    perm = np.random.permutation(nCustomer) + 1 + np.random.choice([0, 1, 2, 3], nCustomer, replace=True)/10
    pointer = 0  # 迭代指针
    lowPointer = 0  # 指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleMile = 0
        i = 0
        back, arae_now = 0,0
        # 当燃油够返回配送中心时，继续装载
        while (pointer <= nCustomer -1) and (vehicleMile < dataDict['MaxMile']):
            vehicleMile -= back
            arae_now=int(math.modf(perm[pointer])[1])
            # arae_now = int()[1]) # 区域索引
            if i == 0:  
                # 未分配任务的无人机第一个任务，计算中心0到这个地方perm[pointer]的距离+这个地方的需求perm[pointer] 
                dis0,dis2 = calmile(perm[pointer],0) # 返回值分别表示到达该区域和从该区域返回
                vehicleMile += dis0 +dataDict['Demand'][arae_now] # 从原点到第一个任务区
            else:
                # 前面已经分配过任务，计算perm[pointer-1]到这个地方perm[pointer]的距离+这个地方的需求perm[pointer]
                dis0,dis1,dis2=calmile(perm[pointer-1],1,perm[pointer])
                vehicleMile += dis1 + dataDict['Demand'][arae_now] # 从上一个任务区到下一个任务区
            # 计算这个地方perm[pointer]到中心0的距离
            back = dis2
            vehicleMile += back
            pointer += 1
            i = i + 1
        # 一旦超过最大航程
        while vehicleMile > dataDict['MaxMile']:
            dis0,dis1,dis2=calmile(perm[pointer-2],1,perm[pointer-1])
            # 这里的pointer - 1是因为上一个循环里有"pointer += 1"
            # 退回最后区域pointer - 1的离开点（back），退回最后区域的进入点（Demand）
            # 退回倒数第二个区域pointer - 2的离开点，加上倒数第二个区域到原点的距离
            vehicleMile = vehicleMile-back-dataDict['Demand'][arae_now]-dis1+calmile(perm[pointer-2],2)
            pointer -= 1
        # print('xxxxxxxxxxxxxx', vehicleMile)
        # 确定最大装载量，即确定了最多可以承担的客户量pointer
        # 在可承受运载量的范围内选择实际的运载量tempPointer
        if lowPointer < pointer:  # lowPointer+1 < pointer
            tempPointer = pointer
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            print("awfully")
            # 一旦进入这个循环，说明某一个地方去不了，则重新生成样本
            perm = np.random.permutation(nCustomer) + 1 + np.random.choice([0, 1, 2, 3], nCustomer, replace=True)/10
            pointer = 0  # 迭代指针
            lowPointer = 0  # 指针指向下界
            permSlice = []
        if pointer == max_num-1:
            permSlice.append(perm[lowPointer::].tolist())
            break
    # 将路线片段合并为染色体
    ind = [0]
    for eachRoute in permSlice:
        ind = ind + eachRoute + [0]
    return ind


'''从染色体解码回路线片段, 每条路径都是以0为开头与结尾'''#评价函数
def decodeInd(ind):
    indCopy = np.array(deepcopy(ind))  # 复制ind，防止直接对染色体进行改动
    idxList = list(range(np.shape(indCopy)[0]))
    zeroIdx = np.asarray(idxList)[indCopy == 0]  # 将结构数据转化为ndarray
    routes = []
    # i表示前一个0，j表示后一个0，共同界定了每个无人机的任务序列
    for i, j in zip(zeroIdx[0::], zeroIdx[1::]):
        if j>i+1: # j=i+1说明是0,0这种情况
            routes.append(ind[i:j]+[0])
        else:
            print("=================出现了0,0这种情况===============")
            print(i,j,zeroIdx)
    return routes



'''辅助函数，返回给定路径的总长度'''
def calRouteLen(routes):
    static_3 = dataDict['distance3']
    T_DIS=[]
    for eachRoute in routes:
        ench_dis=0
        order0 = [int(i) for i in eachRoute]
        order1 = [int(i*10 % 10) for i in eachRoute]
        order2 = 4*(np.array(order0)-1) + np.array(order1)+1 #[ 0 14 22 19  8  7] (6,)    
        order_ga = np.array([max(0,i) for i in order2]) 
        tour_indices = order_ga.reshape(1,-1)
        tour_indices = torch.LongTensor(tour_indices)   
        dis = reward(static_3, tour_indices) 
        dis = dis.numpy()[0]              
        for i in order0:
            dis=dis+dataDict['Demand'][i]
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        ench_dis = dis
        T_DIS.append(ench_dis)
    return T_DIS

'''评价函数，返回解码后路径的总长度，'''
def loadPenalty(T_DIS):
    penalty = 0
    for i in range(len(T_DIS)):
        penalty += 100 * max(0, T_DIS[i] - dataDict['MaxMile'])    
    return penalty

def evaluate(ind):
    routes = decodeInd(ind)  # 将个体解码为路线
    nCities = len(routes)
    T_DIS = calRouteLen(routes)
    total_dis = np.sum(T_DIS)+loadPenalty(T_DIS)+30*nCities
    return (total_dis),

'''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代.交叉操作，选择最好的孩子'''
def genChild(ind1, ind2, nTrail=5):
    hum_op = 0
    routes1 = decodeInd(ind1)  # 将ind1解码成路径
    numSubroute1 = len(routes1)  # 子路径数量
    # 随机选择一个路径，在ind1中随机选择一段子路径subroute1，将其前置
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    # 如果只有一条路径，则需要增加断点变成两条路径
    while numSubroute1<=1:
        hum_op=1
        ind1.insert(np.random.randint(1,len(ind1)-1),0)
        # 重新开始
        routes1 = decodeInd(ind1) 
        numSubroute1 = len(routes1)  # 子路径数量
        # 随机选择一个路径，在ind1中随机选择一段子路径subroute1，将其前置
        subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 提取出区域顺序
    ind1_area = [int(i) for i in ind1] # ind1中的区域顺序
    ind2_area = [int(i) for i in ind2] # ind1中的区域顺序
    subroute1_area = [int(i) for i in subroute1]

    # ind1除subroute1_area外其他区域的重排列
    unvisited = set(ind1_area) - set(subroute1_area)  # 在subroute1中没有出现的区域
    unvisitedPerm = [digit for digit in ind2_area if digit in unvisited]  # 按照在ind2中的顺序排列
    # print(unvisitedPerm)
    # assert type(ind2)==list
    # unvisitedPerm = np.array(unvisitedPerm) + np.random.choice([0, 1, 2, 3], len(unvisitedPerm), replace=True)/10
    # unvisitedPerm = unvisitedPerm.tolist()
    assert type(unvisitedPerm)==list
    for j in range(len(unvisitedPerm)):
        for jj in range(len(ind2)):
            if unvisitedPerm[j]==int(ind2[jj]):
                unvisitedPerm[j]=ind2[jj]
    # print(unvisitedPerm)
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None  # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径，unvisitedPerm中应该包含numSubroute1-1条子路径，
        # 因此断点为numSubroute1-2个
        breakSubroute = []
        if numSubroute1 == 2: #本来有两条子路径，则剩下的这条直接首尾加0即可
            # print("numSubroute1 == 2")
            breakSubroute.append([0] + unvisitedPerm[0:] + [0])
        elif numSubroute1 > 2:
            breakPos = [0]+random.sample(range(1, len(unvisitedPerm)), numSubroute1-2)  # 产生numSubroute1-2个断点
            breakPos.sort()
            for i, j in zip(breakPos[0::], breakPos[1::]):
                breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
            breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        if hum_op==1:#说明进行了操作，需要还原
            breakSubroute=breakSubroute[0][:-1]+breakSubroute[1][1:]
            breakSubroute=[breakSubroute]
        # 评价生成的子路径
        routesFit = np.sum(calRouteLen(breakSubroute)) + loadPenalty(calRouteLen(breakSubroute))
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child+[0]


'''交叉操作'''
def crossover(ind1, ind2):
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2


# 突变操作
def opt(route, k=2):
    # assert type(route)==list
    nCities = len(route)  # 城市数
    optimizedRoute = route  # 最优路径
    minDistance = calRouteLen([route])  # 最优路径长度
    for i in range(1, nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            route_area = [int(i) for i in route] # ind1中的区域顺序
            reversedRoute = route_area[:i]+route_area[i:j][::-1]+route_area[j:]  # 翻转区域后的路径
            # 区域覆盖方向随机选择
            reversedRoute[1:nCities-1] = reversedRoute[1:nCities-1] + np.random.choice([0, 1, 2, 3], nCities-2, replace=True)/10
            # reversedRoute = route[:i]+route[i:j][::-1]+route[j:]  # 翻转后的路径
            # print('reversedRoute',reversedRoute)
            reversedRouteDist = np.sum(calRouteLen([reversedRoute])) + loadPenalty(calRouteLen([reversedRoute]))
            # 如果翻转后路径更优，则更新最优解
            if reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
                optimizedRoute = reversedRoute
    return optimizedRoute

'''用2-opt算法对各条子路径进行局部优化'''
def mutate(ind):
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        optimizedRoute = opt(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child+[0]
    min_dis=evaluate(ind)
    return ind, # min_dis(这个逗号对第一个来说是必须的)
