import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator,algorithms
import deap
from init import init
from pprint import pprint
from init import label,tran_city,global_value


from ge_data.VR import crossover,mutate,evaluate,decodeInd,genInd,calRouteLen,opt

import time
import deap
import numpy as np
import matplotlib.pyplot as plt
from deap import creator,tools,base

import warnings
warnings.filterwarnings('ignore')
class ga_vrp():
    def __init__(self,npop,ngen):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # 最小化问题
        # 给个体一个routes属性用来记录其表示的路线
        creator.create('Individual', list, fitness=creator.FitnessMin)

        # 注册遗传算法操作
        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', evaluate)
        self.toolbox.register('select', tools.selTournament, tournsize=2)
        self.toolbox.register('mate', crossover)
        self.toolbox.register('mutate', mutate)

        # 生成初始族群
        self.toolbox.popSize = npop
        self.pop = self.toolbox.population(self.toolbox.popSize)
        # 遗传算法参数
        self.toolbox.ngen = ngen
        self.toolbox.cxpb = 0.8
        self.toolbox.mutpb = 0.1

        # 记录迭代数据
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register('min', np.min)
        self.stats.register('avg', np.mean)
        self.stats.register('std', np.std)
        self.logbook = tools.Logbook()
        self.logbook.header = 'gen','nevals','avg','std','min','max'
        hallOfFame = tools.HallOfFame(maxsize=1)

        # 遗传算法主程序.(μ + λ)进化算法
        # pop, logbook = algorithms.eaMuPlusLambda(pop, , mu, , , ngen=, stats=stats, halloffame=hallOfFame, verbose=True)# True
        # self.toolbox.evaluate(pop)

    def run(self,i):
        start_time = time.time() 
        invalid_ind = [ind for ind in self.pop]
        fitnesses = map(self.toolbox.evaluate,invalid_ind)
        for ind,fit in zip(self.pop,fitnesses):
            ind.fitness.values = fit

        for g in range(self.toolbox.ngen):
            # offspring = algorithms.varAnd(self.pop, self.toolbox, cxpb=self.toolbox.cxpb, mutpb=self.toolbox.mutpb)
            offspring = algorithms.varOr(self.pop, self.toolbox, lambda_=self.toolbox.popSize, cxpb=self.toolbox.cxpb, mutpb=self.toolbox.mutpb)
            invalids = [ind for ind in offspring]
            fitnesses = map(self.toolbox.evaluate,invalids)
            for ind,fit in zip(offspring,fitnesses):
                ind.fitness.values = fit

            self.pop = self.toolbox.select(self.pop + offspring,k=self.toolbox.popSize)

            record = self.stats.compile(self.pop)
            self.logbook.record(gen=g,nevals=len(self.pop),**record)
            
            if g>0:
                if self.logbook.select('std')[g - 1] < 1*10**(-5):
                    break

        end_time = time . time()  # 记录程序结束运行时间 
        t_run = end_time-start_time
        self.minFit = self.logbook.select('min')
        self.avgFit = self.logbook.select('avg')        
        
        bestInd = tools.selBest(self.pop, k=1)[0]
        # bestInd = hallOfFame.items[0]  # 最好的编码[0, 5.0, 0, 10.0, 9.3, 0, 4.2, 7.2, 0, 1.3, 2.0, 8.3, 6.3, 3.1, 0]
        distributionPlan = decodeInd(bestInd)  # 解码成路径
        bestFit = bestInd.fitness.values  # 路程
        miles = calRouteLen(distributionPlan)

        # print('每个区域的任务需求：')
        # print(Demand)    
        # print('最佳运输计划为：')
        # pprint(distributionPlan)
        # print('最短运输距离为：')
        # print(bestFit)
        # print('各无人机上路程为：')
        # print(miles)

        tour_indices = bestInd.copy()
        # 完整整数路径
        tran_city(tour_indices)
        # 路径标签 小数
        sub_route=label(distributionPlan)
        # 路径标签 整数
        sub_route_i=  [tran_city(i) for i in distributionPlan]
        str_route=label(sub_route_i)
            
        # print('最佳运输路线为：')
        # pprint(tour_indices)
        self.pic(i)
        return tour_indices,miles,str_route,t_run

    def pic(self,i):

        save_path='./pic_ga/'
        fig1, ax1 = plt.subplots()  # , figsize=(20, 8)

        # 画出迭代图
        ax1.plot(self.minFit, 'b-', label='Minimum Fitness')
        ax1.plot(self.avgFit, 'r-', label='Average Fitness')
        ax1.set_xlim([-10,len(self.minFit)])
        ax1.set_xlabel('Gen')
        ax1.set_ylabel('Fitness')
        ax1.legend(loc='best')
        fig1.savefig(save_path+str('Fitness')+str(i)+'.png', bbox_inches='tight', dpi=200)


        # 画出结果
        # plot_pic(a1,0,static,init.lab,tour_indices,miles,sub_route,save_path)
        # tour_indices,miles, route = vrp(a2)
        
        # # 保存结果图
        # fig2.savefig(save_path+'task_assign8.png', bbox_inches='tight', dpi=200)
        # plt.show()

# ga_vrp=ga_vrp(100,400)
# tour_indices,miles,sub_route,t_run = ga_vrp.run()

# ga_vrp.pic()
