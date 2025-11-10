#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: liujie
@software: PyCharm
@file: 自适应遗传算法.py
@time: 2020/11/24 20:42
"""
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator
from ge_data.VR import crossover,mutate,evaluate,decodeInd,genInd,calRouteLen,opt
from init import label,tran_city,global_value
import numpy as np
import matplotlib.pyplot as plt
from deap import creator,tools,base
import random
import time 

save_path="./pic_ga/"

class ga_vrp():
    def __init__(self,npop,ngen):
        # 定义问题
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))   # 单目标优化最小值
        creator.create('Individual', list,fitness = creator.FitnessMin)

        # 定义个体编码
        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', tools.initIterate, creator.Individual, genInd)

        # 创建族群
        self.toolbox.popSize = npop
        self.toolbox.ngen = ngen
        self.toolbox.register('population',tools.initRepeat, list, self.toolbox.individual)
        self.pop = self.toolbox.population(self.toolbox.popSize)

        # 注册所需工具
        self.toolbox.register('evaluate',evaluate)
        self.toolbox.register('select',tools.selTournament,tournsize=2)
        self.toolbox.register('mate',crossover)        # 缺变异几率
        self.toolbox.register('mutate',mutate)  # 缺变异几率
        self.toolbox.register('localOpt',opt)    # 注册2-opt
        self.toolbox.cxpb = 0.8
        self.toolbox.mutpb = 0.3
        # 数据记录
        self.stats = tools.Statistics(key= lambda ind : ind.fitness.values)
        self.stats.register('avg',np.mean)
        self.stats.register('std',np.std)
        self.stats.register('min',np.min)
        self.stats.register('max',np.max)
        self.logbook = tools.Logbook()
        self.logbook.header = 'gen','nevals','avg','std','min','max'


    # 自适应遗传算法
    def GA_improved(self):
        # 开始迭代
        for gen in range(1+self.toolbox.ngen):
            # 配种选择
            print("===============第"+str(gen)+"代种群==================")
            offspring = self.toolbox.select(self.pop,2*self.toolbox.ngen)
            # 复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
            offspring_copy = list(map(self.toolbox.clone,offspring))

            # 变异操作-交叉
            # 计算Pc
            for child1,child2 in zip(offspring_copy[::2],offspring_copy[1::2]):
                # cxpb = self.PC(child1, child2, offspring_copy,self.toolbox.cxpb,self.toolbox.cxpb)
                if random.random() < self.toolbox.cxpb:
                    self.toolbox.mate(child1,child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 变异操作-突变
            for mutant in offspring_copy:
                # mutpb = self.PM(mutant,offspring_copy,self.toolbox.mutpb,self.toolbox.mutpb)
                if random.random() < self.toolbox.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # 对于被改变的个体，重新计算其适应度
            invalid_ind = [ind for ind in offspring_copy if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate,invalid_ind)
            for ind,fit in zip(invalid_ind,fitnesses):
                ind.fitness.values = fit

            # 环境选择-保留精英,保持种群规模100
            pop = tools.selBest(offspring_copy,self.toolbox.popSize,fit_attr='fitness')

            # 对族群中的精英进行优化,也可以对全部个体进行优化
            nOpt = int(self.toolbox.popSize/10)
            pop_opt = tools.selBest(pop,nOpt)
            for ind in pop_opt:
                ind = self.toolbox.mutate(ind)

            # 记录数据
            # compile(sequence)# 将每个注册功能应用于输入序列数据，并将结果作为字典返回
            record = self.stats.compile(pop)
            self.logbook.record(gen=gen,nevals=len(invalid_ind),**record)
            # print(logbook)
            # 当族群适应度的标准差小于1*10^(-9)时，结束运算
            if self.logbook.select('std')[gen - 1] < 1*10**(-5):
                break

        return pop, self.logbook

    # 计算Pc
    def PC(self,child1,child2,pop,k1=1.0,k3=1.0):
        f_ = min(evaluate(child1),evaluate(child2))[0]
        fitness = []
        for route in pop:
            fitness.append(evaluate(route))
        f_min = np.min(fitness)
        f_mean = np.mean(fitness)
        if f_ < f_mean:  # 比较好，则变异率较小
            PC = k1*(f_-f_min) / (f_mean-f_min)
        else:
            PC = k3
        return PC

    # 计算Pm
    def PM(self, mutant,pop,k2=0.5,k4=0.5):
        f_ = evaluate(mutant)[0]
        fitness = []
        for route in pop:
            fitness.append(evaluate(route))
        f_min = np.min(fitness)
        f_mean = np.mean(fitness)
        if f_ < f_mean:
            # print(f_max)
            PM = k2*(f_-f_min) / (f_mean-f_min)
        else:
            PM = k4
        return PM

    def run(self,i):
        start_time = time.time() 
        resultPopGA_improved,logbookGA_improved = self.GA_improved()
        end_time = time . time()  # 记录程序结束运行时间 
        t_run = end_time-start_time
        # print(logbookGA_improved)

        # 绘图——第一张图
        fig1, ax1 = plt.subplots()  
        self.plot(logbookGA_improved,ax1)
        fig1.savefig(save_path+str('Fitness')+str(i)+'.png', bbox_inches='tight', dpi=200)   


        # fig2, ax2 = plt.subplots(1,2, figsize=(20, 8))  # 
        # a1 = ax2[0]
        # a2 = ax2[1]
        bestInd = tools.selBest(resultPopGA_improved,k=1)[0] # 最优路径
        print('遗传算法的最优路径为:'+str(bestInd)+'\n'+'最优路径距离为:'+str(evaluate(bestInd)))
        tour_indices=bestInd.copy()
        tran_city(tour_indices)
        distributionPlan = decodeInd(bestInd)  # 解码成路径
        miles = calRouteLen(distributionPlan) # 路径长度
        sub_route=label(distributionPlan) # 路径标签        
        return tour_indices,miles,sub_route,t_run

        # 第二张图


        # 绘图
        # plot_pic(a1,0,static, init.lab, tour_indices, miles, sub_route, save_path)
        
        # tour_indices,miles,route = vrp(a2)

        # fig2.savefig(save_path+'ga_ge.png', bbox_inches='tight', dpi=200)

        # plt.show()


    # 可视化
    def plot(self, logbookGA_improved,ax1):
        des_pos= global_value.dataDict['des_pos']
        gen = logbookGA_improved.select('gen')
        min = logbookGA_improved.select('min')
        avg = logbookGA_improved.select('avg')
        ax1.plot(gen,min,'r-',label='Minimum Fitness')
        ax1.plot(gen,avg,'b-',label='Average Fitness')
        ax1.set_xlabel('Iteration')
        ax1.set_xlabel('Fitness')
        ax1.legend(loc='upper right')
        ax1.set_title('GA with eliteness preservation strategy iterations, Problem size:{%d}'%des_pos,fontsize = 10)
        plt.tight_layout()




