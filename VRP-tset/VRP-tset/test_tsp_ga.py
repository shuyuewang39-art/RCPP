import numpy as np
import xlwt
from init import init,global_value,label,get_subroute,tran_area,tran_city
data = init(5, 6)

import matplotlib.pyplot as plt
from pic import plot_pic,reward
# from ga_vrp2 import ga_vrp
from ga_vrp import ga_vrp
from greedy_vrp import greedy_vrp,ga_tsp
from tsp_rl import rl_tsp
from ge_data.VR import calRouteLen
from vrp_rl import test_vrp
from Tasks.plot import plot_turn, generate_map, plot_initial
from ge_data.VR import calRouteLen,decodeInd
from ge_data.ga_raw import ga_order,greedy
import warnings
import time
import torch

import warnings
warnings.filterwarnings('ignore')
# 存储数据
cureward = xlwt.Workbook()
sheet_r = cureward.add_sheet(u'reward', cell_overwrite_ok=True)
run_time = xlwt.Workbook()
sheet_t = run_time.add_sheet(u'time', cell_overwrite_ok=True) 

def ga_tsp2(static):
    start_time = time.time() 
    static = torch.Tensor(static)
    order_ga0, dis_ga = ga_order(static[0].data)
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
    return  tour_indices,miles,sub_route,t_run,dis_ga



# 初始化
save_path="./test_pic/pics/"
des_pos= global_value.dataDict['des_pos']

ga_vrp = ga_vrp(100,300)
test_num = 100
best1, best2 = 0,0
for i in range(test_num):
    # 生成一组新的数据集
    data.reset(i)
    all_demand = data.all_demand
    load = global_value.dataDict['MaxMile']
    des_pos = global_value.dataDict['des_pos']
    demand = global_value.dataDict['Demand'] 
    static,l1,x_a,X_val =data.X_VT, data.lab, data.X_a,data.X_val
    tour_indices,miles,task=[],[],[]


    '''获得贪心算法的结果''' #整数完整顺序，子路径距离，整数顺序(str),时间
    tour_indices0,miles0,sub_route0,t_run0 = greedy_vrp(data.X_VT,global_value.dataDict['MaxMile'])

    dis0 = np.sum(miles0)-all_demand
    c_d0 = reward(static, tour_indices0)

    # 存储数据
    tour_indices.append(tour_indices0) # [0, 10, 16, 23, 1, 19, 8, 0]
    miles.append([dis0]) # [xxx,xxx,xx]
    task.append(sub_route0) # ['0, 10, 16, 23, 1, 19, 8, 0']


    '''获得遗传算法的结果'''
    tour_indices1,miles1,sub_route1,t_run1,dis1 = ga_tsp(static)
    dis1 = np.sum(miles1)-all_demand
    c_d1 = reward(static, tour_indices1)
  
    # 存储数据
    tour_indices.append(tour_indices1)
    miles.append([dis1])# miles.append([miles1[0]-all_demand])
    task.append(sub_route1)

    # 获得遗传算法的结果2
    tour_indices2,miles2,sub_route2,t_run2,dis2 = ga_tsp2(static)
    dis2 = np.sum(miles2)-all_demand
    c_d2 = reward(static, tour_indices2)
  
    # 存储数据
    tour_indices.append(tour_indices2)
    miles.append([dis2])# miles.append([miles1[0]-all_demand])
    task.append(sub_route2)

    print(c_d1,c_d2)
    print(t_run1,t_run2)

    # if c_d1<c_d2:
    #     best1=best1+1     
    #     print("\n")
    #     # 
    # elif c_d1>c_d2:
    #     best2=best2+1
    #     print(c_d1,c_d2)
    #     print("\n")
    
    # if i%10==0:
    #     print(best1, best2)
    # 存储数据

    if i<0:
        print("绘图")
        # 绘图
        alos=['GE——','GA——','RL——']
        fig, axes = plt.subplots(1,3, figsize=(24, 10))  # 
        for j in range(3):
            ax = axes[j]
            alo =alos[j]
            tour,mile,task_str=tour_indices[j],miles[j],task[j]
            plot_pic(ax,static,l1,tour,mile,task_str,alo)
        fig.savefig(save_path+'task_assign'+str(i+2)+'.png', bbox_inches='tight', dpi=200)
        # plt.close()
        plt.show()
    
    # 存储时间
    # sheet_t.write(i, 0, t_run0)  # 贪心
    # sheet_t.write(i, 1, t_run1)  # 遗传
    # sheet_t.write(i, 2, t_run)  # 强化
    # # 存储奖励
    # sheet_r.write(i, 0, dis0)  # 贪心
    # sheet_r.write(i, 1, dis1)  # 遗传
    # sheet_r.write(i, 2, tour_len)  # 强化
    #'''

print("test over!")
# cureward.save("reward1.xls")
# run_time.save("time1.xls")

