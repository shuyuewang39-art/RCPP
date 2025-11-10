from init import label,tran_city
import numpy as np
from ge_data.VR import calRouteLen,decodeInd
from ge_data.ga import ga_order,greedy
import warnings
import time
import torch
warnings.filterwarnings('ignore')


def greedy_vrp(static,MaxMile):
    start_time = time.time() 
    static = torch.Tensor(static)
    order_ge, dis_ge = greedy(static[0].data) #[15, 4, 8, 12, 24, 18]
    # 小数完整路径
    order_ga =[0] + [(i+3)//4+(i+3)%4/10 for i in order_ge]  + [0] # [0,4.2, 1.3, 2.3, 3.3, 6.3, 5.1,0]
    # 整数完整路径
    order_ge = [0] + order_ge +[0]
    # 记录程序结束运行时间
    end_time = time . time()   
    t_run = end_time-start_time
    # 拆分为多机
    i,j,dis=1,1,0
    route_f,route_i=[],[]
    while i<= len(order_ga)-1:
        while dis<MaxMile: # 一旦超过了
            i=i+1
            if i > len(order_ga):
                break
            sub = [0]+order_ga[j:i]+[0]
            sub_i = [0]+order_ge[j:i]+[0]
            dis = np.sum(calRouteLen([sub]))
        route_f.append(sub[:-2]+[0])
        route_i.append(sub_i[:-2]+[0])
        i = i-1
        j = i
        dis = 0 
    # print(route)
    miles = calRouteLen(route_f)   

    # 路径标签
    # sub_route=label([order_ge])
    sub_route=label(route_i)
    # 完整小数路径
    tour_indices=[0]
    for i in route_f:
        tour_indices=tour_indices+i[1:]
    # 完整整数路径
    tran_city(tour_indices) 


    return tour_indices,miles,sub_route,t_run


def ga_tsp(static):
    start_time = time.time() 
    static = torch.Tensor(static)
    order_ga0, dis_ga, step = ga_order(static[0].data)
    end_time = time . time()  # 记录程序结束运行时间 
    t_run = end_time-start_time
    # 获得ga算法得到的结果
    order0 = [int(i) for i in order_ga0]
    order1 = [int(i*10 % 10) for i in order_ga0]
    order_ga = 4*np.array(order0) + np.array(order1) + 1 #[ 0 14 22 19  8  7] (6,)

    # 解码成路径
    order_ga1=order_ga0+1.0  #这里加上1.0
    best = [0]+order_ga1.tolist()+[0]
    distributionPlan = decodeInd(best)  
    miles = calRouteLen(distributionPlan)

    # 路径标签
    tour_indices = [0]+order_ga+[0]
    sub_route=label([order_ga])
    
    # # 完整整数路径
    tran_city(order_ga1)
    return  tour_indices,miles,sub_route,t_run,dis_ga,step