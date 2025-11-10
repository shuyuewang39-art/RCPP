import random
import numpy as np
from scipy import spatial
from ge_data.g_data import get_data
import math
from main3 import get_main
from Tasks.g_data2 import get_state

def get_subroute(order):
    indlist = [0]+order.tolist()[0]+[0]
    indCopy = np.array(indlist)

    idxList = list(range(np.shape(indCopy)[0]))
    zeroIdx = np.asarray(idxList)[indCopy == 0]  # 将结构数据转化为ndarray
    routes = []
    # i表示前一个0，j表示后一个0，共同界定了每个无人机的任务序列
    for i, j in zip(zeroIdx[0::], zeroIdx[1::]):
        if i!=j-1:
            routes.append(indlist[i:j]+[0])
    return routes

def label(distributionPlan):
    sub_route=[]
    for sub in distributionPlan:
        list2 = [str(i) for i in sub]  # 使用列表推导式把列表中的单个元素全部转化为str类型
        list3 = ','.join(list2)
        sub_route.append(list3)
    return sub_route

# 整数变小数
def tran_area(indices):
    tour_indices = indices.copy()
    for i in range(len(tour_indices)):
        v = tour_indices[i]
        if v>0:
            v_f=(v+3)//4+(v+3)%4/10
            tour_indices[i]=v_f
    return tour_indices

# 小数变整数
def tran_city(tour_indices):
    for i in range(len(tour_indices)):
        v = tour_indices[i]
        if v>0:
            tour_indices[i]=4*(int(v)-1) + int(np.round(math.modf(v)[0] * 10))+1 
    return tour_indices

class global_value():
    dataDict = {}

class init():
    def __init__(self,uav_num,des_pos):
        self.uav_num = uav_num  # 无人机的最大数量
        self.des_pos = des_pos  # 配送点的数量
        self.reset(0)
    
    def reset(self,i):
        # 生成每个配送点的需求和位置,将配送中心的需求设置为0
        random.seed(i)
        self.demand, self.demand_c  = [0],[]
        # 需求的上下限
        low_d, sup_d  = 50, 150
        for j in range(self.des_pos):
            self.demand.append(random.randint(low_d, sup_d))
            self.demand_c.append(random.randint(low_d, sup_d))

        self.all_demand = np.sum(np.array(self.demand))

        # 生成目标点的位置，外加一个配送中心
        datas = get_data(1, self.des_pos, seed=i)  
        pos, pos_all = datas.ge_area3()  # Y:(sample, city_num, 8)
        self.X_val, self.X_VT, self.X_a = datas.get_state3(pos_all)# X_val (B_val, 4*size+1, 8) 

        city_all = self.X_a[0,:,:] #(size+1, 8)

        self.dis = self.X_VT[0] #[8,4*size+1]
        l0 = np.tile(np.arange(self.des_pos),(4,1)) # (4, area_n)
        self.lab = l0.flatten('F')  # [ 0 ..  0  1 ..  1  2 ... 9 .. 9] (20,)  标签  

        # 用字典存储所有参数 -- 配送中心坐标、顾客坐标;顾客需求、到达时间窗口、服务时间、车型载重量
        # self.dataDict = {}
        global_value.dataDict['MaxMile'] = np.inf # 400600
        global_value.dataDict['NodeCoor'] = city_all #(size+1, 8)
        global_value.dataDict['distance'] = self.dis      #[8,4*size+1]
        global_value.dataDict['distance3'] = self.X_VT      # (B_val, 8, 4*size+1)
        global_value.dataDict['Demand'] = self.demand
        global_value.dataDict['des_pos'] = self.des_pos

    def test_data(self,i):
        # 生成每个配送点的需求和位置,将配送中心的需求设置为0
        random.seed(i)
        self.demand, self.demand_c  = [0],[]
        # 需求的上下限
        low_d, sup_d  = 10, 100
        for j in range(self.des_pos):
            self.demand.append(random.randint(low_d, sup_d))
            self.demand_c.append(random.randint(low_d, sup_d))

        self.all_demand = np.sum(np.array(self.demand))

        # 生成目标点的位置，外加一个配送中心
        # 获得特定测试数据
        locations=[]
        # 初始化城市坐标，input_size表示区域个数，+1表示起点，*4表示每个1区域生成四个点
        start = np.array([[120,5,120,5,120,5,120,5]]) #出发点，起点是第一个点，其坐标位置被固定
        city_task, path_all, area_task, yaw_task = get_main()
        area_task = city_task.reshape(-1,8) #[6,8]
        area_task = np.vstack((start,area_task)) #[7,8]

        one_sample_city = get_state(city_task) # (8,point_city) (8, 24)
        one_sample = np.vstack((start,one_sample_city)) # (8, point_city+1)
        locations.append(one_sample)
        locations=np.array(locations)#   (1, 4*city+1, 8)
        Y_T = locations.transpose(0,2,1) # (1, 8, 4*size+1) 

        self.X_val = locations #   (1, 4*city+1, 8)
        self.X_VT = Y_T  #   (1, 8, 4*city+1)
        self.X_a =  area_task[np.newaxis,:]     #   (1, city+1, 8)

        city_all = self.X_a[0,:,:] #(size+1, 8)

        self.dis = self.X_VT[0] #[8,4*size+1]
        l0 = np.tile(np.arange(self.des_pos),(4,1)) # (4, area_n)
        self.lab = l0.flatten('F')  # [ 0 ..  0  1 ..  1  2 ... 9 .. 9] (20,)  标签  

        global_value.dataDict['MaxMile'] = np.inf # 400
        global_value.dataDict['NodeCoor'] = city_all #(size+1, 8)
        global_value.dataDict['distance'] = self.dis      #[8,4*size+1]
        global_value.dataDict['distance3'] = self.X_VT      # (B_val, 8, 4*size+1)
        global_value.dataDict['Demand'] = self.demand
        global_value.dataDict['des_pos'] = self.des_pos

