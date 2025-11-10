import numpy as np
import torch
import xlrd
import matplotlib.pyplot as plt

def reward(static, tour_indices):
    static = torch.Tensor(static)
    # tour_indices= [int(i) for i in tour_indices]
    tour_indices = torch.tensor(tour_indices,dtype=torch.int64)
    tour_indices = tour_indices.unsqueeze(0)

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


def plot_pic(ax,static,l1,tour_indices,miles,task,alo):
    static = torch.Tensor(static)
    # 所有城市的位置坐标
    i=0
    data_city = static[i, 0:2 ,1:].data  
    ax.scatter(data_city[0], data_city[1], s=100, c=l1, cmap='viridis')

    # 获得访问城市的顺序索引
    idx = torch.tensor(tour_indices.copy(),dtype=torch.int64)  # torch.Size([11])
    idx = torch.unsqueeze(idx, dim=0) # torch.Size([1,11])

    idx = idx.expand(static.size(1), -1)  # torch.Size([8, 20]),复制了8行
    data = torch.gather(static[i].data, 1, idx).cpu().numpy() #(8, 20)，列表示这几个城市
    # 起点的坐标特征，进入点和离开点的坐标
    start = static[i, :, 0].cpu().data.numpy() # (8,)
    x_0, x_1, y_0, y_1 = data[0],data[2],data[1],data[3] # 进入点，离开点的横纵坐标(20,)
    # 所有去过的城市的横纵坐标（已排序）(1, 40)
    x_all = np.vstack((x_0,x_1)).reshape(1,-1,order='F').squeeze() 
    y_all = np.vstack((y_0,y_1)).reshape(1,-1,order='F').squeeze()
    # 所有去过的城市的横纵坐标（已排序，包含出发和返回）(42)
    x = np.hstack((start[0], x_all, start[0])) 
    y = np.hstack((start[1], y_all, start[1]))
    # 访问城市的索引排序，确定其中是0的位置，至少有2个0
    order = np.array(tour_indices).flatten().repeat(2)
    idx = np.hstack((0, order, 0))
    where = np.where(idx == 0)[0]
    u = 0 
    for j in range(len(where) - 1):
        low = where[j]
        high = where[j + 1]

        if low + 1 == high:
            # 中间没有访问城市，则直接跳过，例如[0,0]
            continue
        # 假若相邻两个0的索引分别是3，6则中间访问过4，5，完整的包括【3，4，5，6】
        # ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label='UVA'+str(u)+':'+"order--"+task[u]
        # + "  miles--" + str(np.round(miles[u])))
        ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label="order--"+task[u])
        u=u+1

    ax.scatter(x, y, s=4, c='r', zorder=2)
    ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3) #绘制起点
    xx=np.linspace(0,160,5) # 5个点，说明把160分为4份
    yy=np.linspace(0,200,6) # 6个点，说明把200分为5份
    # xx=np.linspace(0,120,5) # 5个点，说明把160分为4份
    # yy=np.linspace(0,100,5) # 6个点，说明把200分为5份
    # ax.xaxis.set_ticks(xx, fontsize=18)
    # ax.yaxis.set_ticks(yy, fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True)

    my_fontdict = {'family': 'Times New Roman', 'size': 24}
    ax.set_title(alo+"  miles--"+str(np.round(miles[0],2)),  fontdict=my_fontdict)  
    # ax.set_title(alo+"order with distance "+str(np.sum(miles)),fontsize = 16)  
    ax.legend(loc='upper left',prop=my_fontdict)
    # ax.legend()
    ax.set_aspect(1)
    # plt.show()
    # plt.savefig(save_path+'task_assign.png', bbox_inches='tight', dpi=200)


def plot_ga(L_RL,L_GE,save_path,i,step):
    fig, ax = plt.subplots()
    xl = xlrd.open_workbook(r'./best_dis.xls')
    table1 = xl.sheets()[0]
    col0 = table1.col_values(0)[0:step]
    col1 = table1.col_values(1)[0:step]
    col2 = table1.col_values(2)[0:step]

    ax.plot(np.arange(len(col2)), col2, 'r', linestyle='-',label='GA-based')
    ax.plot(np.arange(len(col0)), len(col0) * [L_RL], 'b', linestyle='-', label='RL-based')
    ax.plot(np.arange(len(col0)), len(col0)*[L_GE], 'g', linestyle='-',label='GE-based')

    ax.set_xlabel('iteration', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置x轴名称 x label
    ax.set_ylabel('minimum distance', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置y轴名称 y label
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.legend(loc='upper right')
    plt.savefig(save_path+str(i), bbox_inches='tight', dpi=600)
    plt.show()