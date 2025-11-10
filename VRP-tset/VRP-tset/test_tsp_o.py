import numpy as np
import xlwt
from init import init,global_value,label,get_subroute,tran_area,tran_city
data = init(5, 6)

import matplotlib.pyplot as plt
from pic import plot_pic,reward,plot_ga
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
from tsp_rl import rl_tsp

import warnings
warnings.filterwarnings('ignore')
# 存储数据
cureward = xlwt.Workbook()
sheet_r = cureward.add_sheet(u'reward', cell_overwrite_ok=True)
run_time = xlwt.Workbook()
sheet_t = run_time.add_sheet(u'time', cell_overwrite_ok=True) 


# 初始化
# save_path="./tsp_pic_new/"
des_pos= global_value.dataDict['des_pos']

rl_tsp = rl_tsp(des_pos)
test_num = 500
for ii in range(test_num):
    # 生成一组新的数据集
    data.reset(ii)
    print(f"========={ii}==============")
    all_demand = data.all_demand
    load = global_value.dataDict['MaxMile']
    des_pos = global_value.dataDict['des_pos']
    demand = global_value.dataDict['Demand'] 
    static,l1,x_a,X_val = data.X_VT, data.lab, data.X_a, data.X_val
    tour_indices,miles,task=[],[],[]


    '''获得贪心算法的结果''' #整数完整顺序，子路径距离，整数顺序(str),时间
    tour_indices0,miles0,sub_route0,t_run0 = greedy_vrp(data.X_VT,global_value.dataDict['MaxMile'])

    dis0 = np.sum(miles0)-all_demand
    c_d0 = reward(static, tour_indices0)

    # 存储数据
    tour_indices.append(tour_indices0[1:-1]) # [0, 10, 16, 23, 1, 19, 8, 0]
    miles.append([dis0]) # [xxx,xxx,xx]
    task.append([sub_route0[0][2:-2]]) # ['0, 10, 16, 23, 1, 19, 8, 0']


    '''获得遗传算法的结果'''
    tour_indices1,miles1,sub_route1,t_run1,dis1,step = ga_tsp(static)
    dis1 = np.sum(miles1)-all_demand
    c_d1 = reward(static, tour_indices1)
  
    # 存储数据
    tour_indices.append(tour_indices1)
    miles.append([dis1])# miles.append([miles1[0]-all_demand])
    task.append(sub_route1)

    '''获得单机学习算法的结果'''
    t_run, tour_len, order = rl_tsp.run(data.X_a,data.X_val)
    label_list = [int(i) for i in order[0]]
    sub_route2 = label([label_list])
    tour_rl =[0] + [(i+3)//4+(i+3)%4/10 for i in order[0]]  + [0]
    c_d2 = reward(static, label_list)
    all_demand = data.all_demand
    dis = np.sum(calRouteLen([tour_rl]))

    # 存储数据
    tour_indices.append(label_list)
    miles.append([tour_len])
    task.append(sub_route2)
   
    print(c_d0, c_d1, c_d2)  # 不包含内部距离
    # plot_ga(c_d2,c_d0,save_path,ii,step)

    if ii<100:
        print("绘图")
        # 绘图
        alos=['GE-based','GA-based','RL-based']
        fig, axes = plt.subplots(1,3, figsize=(24, 10))  # 
        for j in range(3):
            ax = axes[j]
            alo =alos[j]
            tour,mile,task_str=tour_indices[j],miles[j],task[j]
            plot_pic(ax,static,l1,tour,mile,task_str,alo)
        fig.savefig('./8task_assign'+str(ii)+'.png', bbox_inches='tight', dpi=800)
        # plt.close()
        plt.show()
    
    # 存储时间
#     sheet_t.write(ii, 0, t_run0)  # 贪心
#     sheet_t.write(ii, 1, t_run1)  # 遗传
#     sheet_t.write(ii, 2, t_run)  # 强化
#     # 存储奖励
#     sheet_r.write(ii, 0, dis0)  # 贪心
#     sheet_r.write(ii, 1, dis1)  # 遗传
#     sheet_r.write(ii, 2, tour_len)  # 强化

#     cureward.save("reward4.xls")
#     run_time.save("time4.xls")


# print("test over!")


