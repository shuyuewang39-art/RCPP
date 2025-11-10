import numpy as np
import torch
from scipy import spatial
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 判断是否结束，是否需要返回原点
# 如果下一个城市是原点，则原点概率为1，否则为0；
# 如果下一个城市不是原点，则可以去的地方（有需求，而且需求小于负载）概率为1，否则为0
# 如果没有负载，则必须回到原点；如果所有需求都满足了，也必须回到原点。
def update_mask(mask, dynamic,chosen_idx=None, dis_all=None,static_city=None):
    # dynamic为(batch_size, feats, num_cities),chosen_idx表示选择城市的索引
    """Updates the mask used to hide non-valid states.
    Parameters----------dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
    """
    dis_all = torch.from_numpy(dis_all)
    # Convert floating point to integers for calculations获得负载和需求
    loads = dynamic.data[:, 0]  # (batch_size, num+1)
    loads = torch.repeat_interleave(loads, 4, dim=1)[:,3:]
    demands = dynamic.data[:, 1]  # (batch_size, num+1)
    size, batch = static_city.size(2),static_city.size(0)

    # If there is no positive demand left, we can end the tour.
    # Note that the first node is the depot, which always has a negative demand
    # 如果所有的需求都满足了，就可以结束训练
    if demands.eq(0).all(): # 检查所有键是否为真
        return demands * 0.

    # 完成这个任务所需的全部负载
    index1 = chosen_idx.unsqueeze(1).repeat(1,size).unsqueeze(1) # [b,1,4*size+1]
    index2 = torch.zeros((batch,size,1),dtype=torch.int64)#.cuda() # (batch,4*size,1)
    # index1=torch.tensor([[[0,0,0]],[[1,1,1]]]) torch.Size([b, 1, 4*size+1]
    reach = torch.gather(dis_all, dim=1,index=index1) # torch.Size([B, 1, 4*size+1]
    back = torch.gather(dis_all, dim=2, index=index2) # torch.Size(batch,size,1)
    reach = reach.squeeze(1) # [B, size]
    back = back.squeeze(2)   # [B, size]
    demands_city = torch.repeat_interleave(demands, 4, dim=1)[:,3:]
    task = demands_city + reach + back  # (batch_size, num+1)
    task[:,0] = 0

    # 这一步返回哪些地方可以去，哪些地方不可以去：这个地方有需求，而且需求小于负载
    new_mask = demands_city.ne(0) * task.lt(loads) # (batch_size, num+1)可以去的地方是1，其他是0

    # We should avoid traveling to the depot back-to-back
    # 这一步返回chosen_idx>0的地方，去的地方中是否是原点
    repeat_home = chosen_idx.ne(0) # batch_size中返回原点的样本

    # 如果选择的这些地方存在原点，下一个位置
    if repeat_home.any():
        # 这次如果选择的是城市，则下一次可以选原点
        new_mask[repeat_home.nonzero(), 0] = 1.
    # 如果选择的这些地方全都是原点，那么下一个地方就不可以选择原点
    if (~repeat_home).any():
        new_mask[(~repeat_home).nonzero(), 0] = 0.

    has_no_load = loads[:, 0].eq(0).float() # 没有负载的batch为1，其余为0
    has_no_demand = demands[:, 1:].sum(1).eq(0).float() # 没有需求的batch为1，其余为0
    # 获得要么没有需求要么没有负载的batch，这种情况需要返回原点获得新的负载，或者回到原点结束本次任务
    combined = (has_no_load + has_no_demand).gt(0)
    if combined.any(): # 如果全为空、0、False，则返回False
        new_mask[combined.nonzero(), 0] = 1. # 结束的话，应该返回原点
        new_mask[combined.nonzero(), 1:] = 0. 

    return new_mask.float()


def get_index(i): #i,<class 'numpy.ndarray'>
    if i.ndim==0:
        i=np.array([i])
    # 更新那些即将访问的城市的绑定城市的需求和当前的负载      
    area_idex = (i-1)//4  # 获得选择的城市归属区域的索引(从0开始)   
    point_idx=(i-area_idex*4)%5 # 获得选择的城市归属区域的方向的索引([1,2,3,4])
    index=np.array([1,2,3,4])
    index_all = np.tile(index,(np.shape(point_idx)[0],1)) # (220, 4) 所有方向
    index_oth = np.zeros((np.shape(point_idx)[0],3)) # (220, 3) 其他方向
    for j in range(np.shape(point_idx)[0]):
        index_oth[j] = np.delete(index_all[j,:], np.where(index_all[j,:] == point_idx[j])) 
    area_num =  np.tile(area_idex*4,3).reshape(np.shape(point_idx)[0],3,order='F') # (220, 3)
    same_visit_idx = area_num + index_oth   # (220, 3) 
    return  same_visit_idx



# 这个函数会根据网络的输出更新动态元素
# 对于负载而言：如果回到原点，则负载为1；如果访问城市，则负载更新
# 对于需求而言：如果回到原点，则原点需求为0；如果访问城市，则其需求更新，而且原点的需求表示的是当前还剩多少负载,是一个负数
def update_dynamic(dynamic, chosen_idx, last_idx,static_city,max_load):
    # [b,2,s+1], [b], [b,1]

    # chosen_idx中大于0的是要访问的城市visit，等于0的代表出发点depot
    visit = chosen_idx.ne(0) # torch.Size([batch_size]),True or false
    depot = chosen_idx.eq(0) # torch.Size([batch_size])

    # Clone the dynamic variable so we don't mess up graph
    all_loads = dynamic[:, 0].clone() # (batch_size, num+1)
    all_demands = dynamic[:, 1].clone() # (batch_size, num+1)

    # 获得网络选择城市的负载和需求
    chosen_area = ((chosen_idx+3)//4).unsqueeze(1)  # (batch_size, 1)    
    load = torch.gather(all_loads, 1, chosen_area)  # (batch_size, 1)
    demand = torch.gather(all_demands, 1, chosen_area) # (batch_size, 1)

    # 获得到达区域点的消耗
    tour_indices = torch.cat((last_idx, chosen_idx.unsqueeze(1)), 1)	# [batch_size, 2]
    # print(static_city[0,0:4,:])
    dis0, dis1 = load_inter(static_city, tour_indices) # dis0返回原点，dis1没返回原点
    dis1 = dis1.unsqueeze(1)
    dis1 = torch.tensor(dis1, dtype=torch.float32)/(max_load) 
    # 存在要访问的城市
    if visit.any(): 
        #  将输入input张量每个元素的夹紧到区间 [min,max]
        new_load = torch.clamp(load - demand - dis1, min=0)
        # 由于mask的存在，负载一定大于需求时，可以完全满足，则new_demand=0
        new_demand = torch.clamp(demand - load, min=0)
        # 用于输出数组的非零值的索引,即要访问的城市的索引
        visit_idx = visit.nonzero().squeeze()  # torch.Size([batch0]) tensor([0,..., 1])
        # 同区域的其他点的索引，应当将其需求改为0
        i = chosen_idx[visit_idx].cpu().numpy() # 获得选择的城市索引(不包含原点) 
        area_i = (i+3)//4  # 获得选择的区域索引(不包含原点) 
        # same_visit_idx= self.get_index(i)  # 返回一个（batch0,3)的数组，行表示样本，列表是其他城市的索引

        # 更新那些即将访问的城市的需求和当前的负载
        all_loads[visit_idx] = new_load[visit_idx]
        all_demands[visit_idx, area_i] = new_demand[visit_idx].view(-1)
        all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)# 原点的需求等于剩余的负载，在前面加上负号

    # 存在返回原点的Return to depot to fill vehicle load
    if depot.any():
        # 返回原点后，新的负载变为1，原点的需求是0
        all_loads[depot.nonzero().squeeze()] = 1. # torch.Size([256, 11])
        all_demands[depot.nonzero().squeeze(), 0] = 0.
    
    tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
    return torch.tensor(tensor.data, device=dynamic.device)

def load_inter(static, tour_indices):
    # static:torch.Size([256, 8, 41]) tour_indices:torch.Size([256, step]),step表示选择了多少步
    idx = tour_indices.unsqueeze(1).expand(-1, int(static.size(1)/2), -1)   #将tour_indices复制了 torch.Size([256, 4, 13])
    # tour按照这一回合选择的城市的索引获得其二维坐标按顺序排列的的结果
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # torch.Size([256, step, 4])  
    # start表示每个样本的起始位置的坐标torch.Size([256, 1，8])
    start = static.data[:, 0:2, 0] # torch.Size([256, 2])
    start = torch.cat([start,start],dim=1).unsqueeze(1) #torch.Size([256, 1, 4])
    # y表示一个完整来回的位置序列表示torch.Size([256, 1+step+1，4])
    y1 = torch.cat((tour, start), dim=1) #按维数1（列）拼接，行不变列变（0，1，...5,0）
    y0 = tour
    tour_len1 = torch.sqrt(torch.sum(torch.pow(y1[:, :-1, 2:4] - y1[:, 1:,0:2], 2), dim=2))
    tour_len0 = torch.sqrt(torch.sum(torch.pow(y0[:, :-1, 2:4] - y0[:, 1:,0:2], 2), dim=2))
    return tour_len1.sum(1), tour_len0.sum(1)

def reward(static, tour_indices):
    # static:torch.Size([256, 8, 41]) tour_indices:torch.Size([256, step]),step表示选择了多少步
    # idx表示每个样本在这一回合选择的城市的索引的排序
    # print(tour_indices)
    idx = tour_indices.unsqueeze(1).expand(-1, int(static.size(1)/2), -1)   #将tour_indices复制了 torch.Size([256, 4, 13])
    # tour按照这一回合选择的城市的索引获得其二维坐标按顺序排列的的结果
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # torch.Size([256, step, 4])  
    # start表示每个样本的起始位置的坐标torch.Size([256, 1，8])
    start = static.data[:, 0:2, 0] # torch.Size([256, 2])
    start = torch.cat([start,start],dim=1).unsqueeze(1) #torch.Size([256, 1, 4])
    # y表示一个完整来回的位置序列表示torch.Size([256, 1+step+1，4])
    y = torch.cat((start, tour, start), dim=1) #按维数1（列）拼接，行不变列变（0，1，...5,0）
    a=torch.pow(y[:, :-1, 2:4] - y[:, 1:,0:2], 2) #[b,step,2]
    dis = torch.sum(a, dim=2) #[b,step]
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1, 2:4] - y[:, 1:,0:2], 2), dim=2))
    return tour_len.sum(1)





