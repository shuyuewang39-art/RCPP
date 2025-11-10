import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
from Models.actor import DRL4TSP
from Models.critc import StateCritic
import matplotlib
import matplotlib.pyplot as plt

from Tasks.vrp import update_dynamic,update_mask,reward

matplotlib.use('Agg')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))
import warnings
warnings.filterwarnings("ignore")
pic=0


def compute_dis(static, tour_indices,demand):# [step]
    static, tour_indices = static.unsqueeze(0), tour_indices.unsqueeze(0)
    idx = tour_indices.unsqueeze(1).expand(-1, int(static.size(1)/2), -1)   #将tour_indices复制了 torch.Size([256, 4, 13])
    # tour按照这一回合选择的城市的索引获得其二维坐标按顺序排列的的结果
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # torch.Size([256, step, 4])  
    start = static.data[:, 0:2, 0] # torch.Size([256, 2])
    start = torch.cat([start,start],dim=1).unsqueeze(1) #torch.Size([256, 1, 4])
    # y表示一个完整来回的位置序列表示torch.Size([256, 1+step+1，4])
    y = torch.cat((start, tour, start), dim=1) #按维数1（列）拼接，行不变列变（0，1，...5,0）
    dis = torch.sum(torch.pow(y[:, :-1, 2:4] - y[:, 1:,0:2], 2), dim=2) # [b,step+1]

    # 找到0元素
    order=np.array([0]+tour_indices[0].tolist()+[0])
    area_order=(order+3)//4
    where = np.where(order == 0)[0]
    route_dis,area_dis=[],[]
    for j in range(len(where) - 1):
        d=0
        low = where[j]
        high = where[j + 1]
        if low + 1 == high:# 中间没有访问城市，则直接跳过，例如[0,0]
            continue
        sub=dis[0][low:high]
        a=torch.sum(torch.sqrt(sub))
        route_dis.append(a.tolist())
        for i in area_order[low:high]:
            d=d+demand[i]
        area_dis.append(100*max_load*d.tolist())
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1, 2:4] - y[:, 1:,0:2], 2), dim=2))
    all_sub_dis=np.array(route_dis)+np.array(area_dis)
    return tour_len.sum(1),route_dis,area_dis,all_sub_dis

#def plot(static, tour_indices,l1,dynamic,batch_idx,epoch=None):
    """Plots the found solution.""" 
    global pic
    pic=pic+1
    pic_dir = './tes_pic/'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    plt.close('all')
    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 223 else 1
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,sharex='col', sharey='row')
    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]
    color=['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w','violet','plum','navy']
    for i, ax in enumerate(axes):
        # 记录所有城市的坐标位置
        data_city = static[i, 0:2 ,1:].data.cpu()  # 所有城市的位置坐标
        demand = dynamic.data[i, 1]  # (1, num+1)
        j=0

        for ii in range(0,data_city.size(1),4):# l1[ii:ii+4]
            plt.scatter(data_city[0][ii:ii+4], data_city[1][ii:ii+4], s=100, c=color[j], cmap='viridis',label='intra_dis:'+str(round(100*max_load*demand[j+1].tolist())))
            j=j+1
        # 获得访问城市的顺序索引，以及位置信息
        idx = tour_indices[i]  # 样本i选择的城市索引，torch.Size([1, 20])
        total_dis,route_dis,area_dis,all_sub_dis = compute_dis(static[i], idx, demand)
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)
        idx = idx.expand(static.size(1), -1)  # torch.Size([8, 20]),复制了8行
        data = torch.gather(static[i].data, 1, idx).cpu().numpy() #(8, 20)，列表示这几个城市
       
        # 起点的坐标特征，进入点和离开点的坐标
        start = static[i, :, 0].cpu().data.numpy() # (8,)
        x_0,x_1,y_0,y_1 = data[0],data[2],data[1],data[3] # 进入点，离开点的横纵坐标(20,)
        
        # 所有去过的城市的横纵坐标（已排序）(1, 40)
        x_all = np.vstack((x_0,x_1)).reshape(1,-1,order='F').squeeze() 
        y_all = np.vstack((y_0,y_1)).reshape(1,-1,order='F').squeeze()

        # 所有去过的城市的横纵坐标（已排序，包含出发和返回）(42)
        x = np.hstack((start[0], x_all, start[0])) 
        y = np.hstack((start[1], y_all, start[1]))
        
        # 访问城市的索引排序，确定其中是0的位置，至少有2个0
        # 因为位置信息包含入点和出点，所以相应其标签也应该复制成两个
        order = tour_indices[i].cpu().numpy().flatten().repeat(2)  
        idx = np.hstack((0, order, 0))
        where = np.where(idx == 0)[0]
        num = 0
        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                # 中间没有访问城市，则直接跳过，例如[0,0]
                continue
            # 假若相邻两个0的索引分别是3，6则中间访问过4，5，完整的包括【3，4，5，6】
            dis=route_dis[num]
            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label='inter_dis:'+str(round(dis)))
            num=num+1

        ax.legend(loc="upper right", fontsize=6, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=6)

        xx=np.linspace(0,160,5) # 5个点，说明把120分为4份
        ax.xaxis.set_ticks(xx)
        ax.yaxis.set_ticks(xx)
        ax.grid(True)
        all_sub_dis=np.array([round(i) for i in all_sub_dis])
        ax.set_title("order with distance "+str(all_sub_dis),fontsize = 14)
        # ax.grid()
        # ax.set_xlim(-100, 100)
        # ax.set_ylim(-100, 100)

    plt.tight_layout()
    plt.savefig(pic_dir+"_"+str(pic), bbox_inches='tight', dpi=200)


def test_vrp(X_a,X_val,load,area_n,demands,L):
    # 这一部分设置了最大负载，最大需求，静态量，动态量
    STATIC_SIZE = 8   # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)
    hidden_size = 256
    num_layers = 2
    dropout = 0.1
    global max_load
    max_load = load/100

    print('Starting VRP training')

    dynamic_shape = (1, 1, area_n + 1)
    loads = torch.full(dynamic_shape, 1.) # 全是1
    
    demands = np.array(demands)[np.newaxis,:]/load # 归一化后的需求
    demands = demands.reshape(1,1,-1)


    dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))  #[1，2，size+1]

    locations = torch.from_numpy(X_a)
    locations = locations.transpose(1, 2)


    # 生成每个样本中，这些城市之间的距离
    dis_all=[]
    city_pos0=X_val[:,:,0:2]  # [B, 4*size+1, 0:2]
    city_pos1=X_val[:,:,2:4]  # [B, 4*size+1, 2:4]
    for j in range(np.shape(city_pos0)[0]):
        array_e = city_pos0[j]  # [4*size+1, 0:2]
        array_s = city_pos1[j]  # [4*size+1, 2:4]
        dis = spatial.distance.cdist(array_s, array_e, metric='euclidean')   
        dis_all.append(dis)
    dis_all=np.array(dis_all)/load  

    static = locations
    static_city = torch.from_numpy(X_val)


    # 初始化网络模型 
    actor = DRL4TSP(STATIC_SIZE,DYNAMIC_SIZE,hidden_size,update_dynamic,update_mask,num_layers,dropout).to(device)
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, hidden_size).to(device)

    # 加载网络参数
    print("加载网络参数")

    path1 = "./Models/actor0.pt"
    actor.load_state_dict(torch.load(path1, device))

    path2 = "./Models/critic0.pt"
    critic.load_state_dict(torch.load(path2, device))

    # 训练结束后，会进行测试，首先生成测试集数据，然后进行测试
    print("开始对测试集数据进行验证")
    actor.eval()

    static_city = static_city.transpose(1, 2)

    static = static.to(device)
    dynamic = dynamic.to(device)

    start_time = time.time() 
    with torch.no_grad():
        tour_indices, _ = actor.forward(static, dynamic, static_city, dis_all, load)
    # 记录程序结束运行时间
    end_time = time . time()   
    t_run = end_time-start_time
    cost = reward(static_city, tour_indices).mean().item()

    print('Average tour length: ', cost)
    return np.mean(cost),tour_indices,t_run

    
