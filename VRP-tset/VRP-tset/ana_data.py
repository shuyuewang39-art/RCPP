#!/usr/bin/python3
#code-python(3.6)
import matplotlib.pyplot as plt
import xlrd
import numpy as np
# xl = xlrd.open_workbook(r'./time1.xls')
# # xl = xlrd.open_workbook(r'./reward1.xls')
# table = xl.sheets()[0]
# col0 = table.col_values(0) #list
# col1 = table.col_values(1)
# col2 = table.col_values(2)


# fig,axes = plt.subplots(1,2)
# ax1=axes[0] 
# ax2=axes[1] 
# # col0 = [np.log(i) for i in col0]
# ax1.boxplot((col0,col2),labels=('Greedy', 'RL'),
# medianprops={'color': 'red', 'linewidth': '1.5'},
# meanline=True,showmeans=True,
# meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
# flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 10},
# showfliers=False)

# ax1.boxplot((col0,col2),labels=('Greedy', 'RL'),            
#             patch_artist=True,
#             boxprops={'color': '#ffff00', 'facecolor': '#0066ff'},
#             capprops={'color': '#ff3333', 'linewidth': 2},
#             showmeans=True,
#             meanline=True,
#             showfliers=False)
# plt.boxplot((col1,col2),labels=('GA', 'RL'))
# fig = plt.figure()  # abels='GA', 
# ax2.boxplot((col1,),labels=('GA',) ,         
#             patch_artist=True,
#             boxprops={'color': '#ffff00', 'facecolor': '#0066ff'},
#             capprops={'color': '#ff3333', 'linewidth': 2},
#             showmeans=True,
#             meanline=True,
#             showfliers=False)
# ax1.tick_params(labelsize=16)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]

# fig = plt.figure() 
# plt.boxplot((col0,col1,col2),labels=('Greedy','GA', 'RL'),            
#             patch_artist=True,
#             boxprops={'color': '#ffff00', 'facecolor': '#0066ff'},
#             capprops={'color': '#ff3333', 'linewidth': 2},
#             showmeans=True,
#             meanline=True,
#             showfliers=False)
# # plt.boxplot()
# fig = plt.figure()  
# ax1 = fig.add_subplot(111)  
# x = np.arange(0,len(col1))  
# d_r = np.array(col2)
# d_r1 = np.array(col1)
# ax1.set_title('Scatter Plot')  
# ax.set_xlabel('train_steps', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置x轴名称 x label
# ax.set_ylabel('loss', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置y轴名称 y label
# plt.tick_params(labelsize=14)
# labels = ax1.get_xticklabels() + ax1.get_yticklabels()
# [label.set_fontname('Times New Roman') for label in labels]
# plt.xlabel('X')  
# plt.ylabel('Y')  
# ax1.plot(x,d_r, '-r' ,label = 'diff')

# ax1.scatter(x,d_r, c = 'r',marker = 'o' ,label = 'diff')  
# ax1.scatter(x,col2, c = 'b',marker = 'o' , label ='RL')  

# plt.legend()
# plt.show()  

from numpy import *
import numpy as np
import xlrd
import matplotlib.pyplot as plt


def plot_ga():
    fig, ax = plt.subplots()
    xl = xlrd.open_workbook(r'C:/Users/L/Desktop/cover_tasks/best_dis.xls')
    table1 = xl.sheets()[0]
    col0 = table1.col_values(0)
    col1 = table1.col_values(1)
    col2 = table1.col_values(2)

    ax.plot(np.arange(len(col2)), col2, 'r', linestyle='-',label='GA-based')
    ax.plot(np.arange(len(col0)), len(col0)*[230.6980], 'b', linestyle='-',label='RL-based')


    # ax.plot(np.arange(len(col0)), col0, 'b', linestyle='-',label='Average distance')
    # ax.plot(np.arange(len(col1)), col1, 'c', linestyle='-',label='variance of distance')
    ax.set_xlabel('iteration', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置x轴名称 x label
    ax.set_ylabel('minimum distance', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置y轴名称 y label
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.legend(loc='upper right')
    plt.savefig('./win4', bbox_inches='tight', dpi=600)


def plot_reward():
    fig, ax = plt.subplots()
    xl = xlrd.open_workbook(r"C:/Users/L/Desktop/tsp3/reward3.xls")
    table1 = xl.sheets()[0]
    col1 = table1.col_values(0)
    n = 1
    m = math.ceil((len(col1)-0)/n)
    labelA = np.arange(m)
    j = 0
    print(len(col1))
    for i in range(0, len(col1)-0, n):
        labelA[j] = float(col1[i])
        j = j+1
    ax.plot(np.arange(len(labelA)), labelA, 'r', linestyle='-')
    ax.set_xlabel('train_steps', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置x轴名称 x label
    ax.set_ylabel('reward', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置y轴名称 y label
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.savefig('./train_re_u3', bbox_inches='tight', dpi=600)

def plot_loss():
    fig, ax = plt.subplots()
    xl = xlrd.open_workbook(r"C:/Users/L/Desktop/tsp3/loss3.xls")
    
    col=[]
    ave_loss=[]
    for j in range(0,3):
        table1 = xl.sheets()[j]
        print(table1.ncols)
        for i in range(table1.ncols):
            a = [float(num) for num in table1.col_values(i)]
            ave = sum(a) / len(a)
            ave_loss.append(ave)
    # print(len(col1))
    col1=ave_loss #col1=col
    n = 1
    m = math.ceil((len(col1)-0)/n)
    labelA = np.arange(m)
    j=0
    for i in range(0, len(col1)-0, n):
        labelA[j] = float(col1[i])
        j = j+1
    ax.plot(np.arange(len(labelA)), labelA, 'r', linestyle='-')
    ax.set_xlabel('train_steps', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置x轴名称 x label
    ax.set_ylabel('loss', fontdict={'family' : 'Times New Roman', 'size':20})  # 设置y轴名称 y label
    plt.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.savefig('./train_lo_u3', bbox_inches='tight', dpi=600)


def test_data():
    xl = xlrd.open_workbook('./reward4.xls')
    table = xl.sheets()[0]
    col0 = table.col_values(0) 
    col1 = table.col_values(1)
    col2 = table.col_values(2)
    best0,best1,best2=0,0,0
    best01,best02,best12=0,0,0
    best123=0
    a1 = 0
    print(len(col0))
    
    a = [round(float(num),1) for num in col0]
    b = [round(float(num),1)  for num in col1]
    c = [round(float(num),1)  for num in col2]
    ave0,ave1,ave2 = sum(a) / len(a),sum(b) / len(b),sum(c) / len(c)
    std0,std1,std2 = np.std(a),np.std(b),np.std(c)
    for i in range(len(col0)):
        re3 = np.array([a[i],b[i],c[i]])
        index_min = np.argmin(re3)
        if index_min==0:
            a1=a1+1

        if index_min==0:
            if re3[1]==re3[2] and re3[1]==re3[0]:
                best123=best123+1         
            elif re3[1]==re3[0]:
                best01=best01+1
            elif re3[2]==re3[0]:
                best02=best02+1 
            else:
                best0=best0+1           
        elif index_min==1:
            if re3[1]==re3[2]:
                best12=best12+1
            else:
                best1=best1+1
        else:
            best2=best2+1

    print(best0,best1,best2)
    print(best01, best02, best12, best123, a1)
    # best12：2，3算法都赢
    # best123：1，2，3算法都赢
    # a1 best1+best12+best123+best01，best12+best123+best2

    print(ave0,ave1,ave2)
    print(std0, std1, std2)
    r1, r2, r3 = a1, best1 + best12 + best123 + best01, best12 + best123 + best2
    print(r1, r2, r3)
    print(r1/len(col0)*100,r2/len(col0)*100,r3/len(col0)*100)
    




if __name__ == '__main__':
    test_data()
    # plot_ga()
    # plot_loss()
    # plot_reward()