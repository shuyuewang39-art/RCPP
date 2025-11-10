import numpy as np
import torch
from gpn import GPN
import time
import warnings
warnings.filterwarnings('ignore')

class rl_tsp():
    def __init__(self,size):

        self.size = size
        model_root = "./model/gpn_tsp9/" 
        # model_root = "./model/gpn_tsp8/"'21.pt'
        # model_root ='./model/gpn_tsp6/'

        self.model = GPN(n_feature=8, n_hidden=128).cuda()
        self.model = torch.load(model_root + "498_1499.pt") # load model1583.pt
        # self.model = torch.load(model_root + "1583.pt")

    def run(self, X_CITY, X_val):
        X_val = torch.Tensor(X_val).cuda()
        X_CITY = torch.Tensor(X_CITY).cuda()
        X, X_0 = X_CITY.reshape(-1,self.size+1,8),X_val.reshape(-1,self.size*4+1,8)
        ge_start_time = time.time() # 记录程序开始运行时间
        # mask 设置
        mask = torch.zeros(1,self.size+1).cuda()
        mask[:,0] = -np.inf
        order = np.zeros((1,1))
        # 奖励 设置
        R = 0
        reward = 0
        tour_len = 0
        # 初始化
        h = None
        c = None            
        Y = X_0
        x = Y[:,0,0:4]
        # 用于计算奖励，分别表示起点和终点
        Y0 = Y[:,0,:]  
        Y_ini = Y[:,0,:]
        for k in range(self.size):
            # 根据网路模型得到下一个访问城市
            output, h, c, hidden_u = self.model(x=x, X_all=X, h=h, c=c, mask=mask)
            idx = torch.argmax(output, dim=1)
            # 存储访问顺序
            order=np.hstack((order,idx.cpu().reshape(-1,1)))
            # 获得访问城市的位置坐标
            Y1 = Y[[i for i in range(1)], idx.data]
            # 计算奖励
            reward = torch.norm(Y1[:,0:2]-Y0[:,2:4], dim=1)
            R += reward
            # 更新上一个城市的位置
            Y0 = Y1.clone()
            x = Y[[i for i in range(1)], idx.data]
            x = x[:,0:4] #[B,4]只需要8维状态的前4维
            
            # 选择非原点的城市的样本索引
            area_num=((idx.cpu().data).numpy()-1)//4+1
            mask[[j for j in range(1)], area_num] += -np.inf

        order = np.array(order)[:,1:]    #(128, 7)
        ge_end_time = time . time()  # 记录程序结束运行时间 
        t_run = ge_end_time-ge_start_time

        R += torch.norm(Y1[:,2:4]-Y_ini[:,0:2], dim=1)
        tour_len += R.mean().item()
        print('validation tour length:', tour_len)
        return t_run, tour_len, order





    '''# 获得特定测试数据
    locations=[]
    # 初始化城市坐标，input_size表示区域个数，+1表示起点，*4表示每个1区域生成四个点
    start = np.array([[120,5,120,5,120,5,120,50]]) #出发点，起点是第一个点，其坐标位置被固定
    city_task,path_all,area_task,yaw_task = get_main()
    one_sample_city = get_state(city_task) # (8,point_city) (8, 24)
    one_sample = np.vstack((start,one_sample_city)) # (8,point_city+1)
    locations.append(one_sample)
    locations=np.array(locations)#   (1, 8, point_city+1)
    locations=torch.from_numpy(locations)  # (1, 4*size+1,8)
    locations = locations.float()
    Y_T = locations.transpose(2,1) # (1, 8, 4*size+1) 
    test(locations,Y_T)'''





