# import matplotlib.pyplot as plt
# import numpy as np
# import math
# CROSS_RATE = 0.1
# MUTATE_RATE = 0.02
# POP_SIZE = 500
# N_GENERATIONS = 200
# pi = 3.1415926
# def get_yaw(end_pos,star_pos):
#     theta = np.arctan((end_pos[1]-star_pos[1])/(end_pos[0]-star_pos[0])) if star_pos[0] != end_pos[0] else -pi/2# -pi
#     if end_pos[0] < star_pos[0]:  # 在二三象限
#         theta = theta+pi
#     end_yaw = theta
#     return end_yaw

# def get_ga_in(area_num,static):
#     # static:torch.Size([batch, 8, area_num*4+1])-torch.Size([8, area_num*4+1])
#     area_task = [[] for _ in range(area_num)]
#     for i in range(0, area_num): 
#         for j in range(4):
#             area_task[i].append([static[0:2,i*4+j+1], static[2:4,i*4+j+1]])
#     return area_task


# def choose_path(area_copy, start, mask):
#     #(6, 4, 2)print(np.shape(area_copy))
#     # 根据无人机当前的位置选择一个最近的航道
#     length_all = []
#     m = float('inf')
#     for j in range(len(area_copy)):
#         if j in mask:
#             for i in range(4):
#                 length_all.append(m)
#             continue
#         area_j = area_copy[j]   #第j个区域（4，2，2）
#         for i in range(4):
#             area_j_i = area_j[i] # 第j个区域的第i个方向（2，2）
#             entry = area_j_i[0]
#             path_length = np.linalg.norm(entry.numpy()-np.array(start))
#             length_all.append(path_length)
#     index_min = np.argmin(length_all)
#     # print(index_min)
#     dis = length_all[index_min]
#     area_index = int(index_min / 4) # 获得最近的区域
#     path_index = index_min % 4 # 获得最近的方向
#     path_num = area_index+path_index/10  # 整数部分表示区域，小数部分表示方向，用于遗传编码
#     return index_min,dis,area_index,path_index


# def greedy(static_raw):# torch.Size([8, 25])
#     static = static_raw.clone()
#     area_num = int((np.shape(static)[1]-1)/4)
#     start = static[0:2,0]
#     area = get_ga_in(area_num,static) # (area_num,4,2,2)
#     start0 = (start.numpy()).copy()  
#     start_position = (start.numpy()).copy()
#     area_copy = area.copy()
#     dis_length = 0
#     greedy_order = []
#     mask = [10]
#     for i in range(0, area_num):
#         # 根据当前位置选择一个最近的航道,绘制到达区域的路径，area_copy会被改变，删去已经选择的区域
#         index_min, dis, area_index, path_index = choose_path(area_copy, start_position,mask)
#         greedy_order.append(index_min+1)
#         # print(index_min)
#         # print("area_index:"+str(area_index))
#         mask.append(area_index)
#         # print("mask:"+str(mask))
#         # 计算距离
#         dis_length = dis_length + dis
#         # 搜索完后，无人机的位置需要更新
#         leave = area_copy[area_index][path_index][1]         #[2,2]
#         start_position = [leave[0], leave[1]]
#     # 返回原点
#     start_position = [i.numpy()for i in start_position]
#     start_position = np.array(start_position)
#     path_length = np.linalg.norm(start_position-start0)
#     dis_length = dis_length + path_length
#     return greedy_order,dis_length



# class GA(object):
#     def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
#         self.DNA_size = DNA_size
#         self.cross_rate = cross_rate
#         self.mutate_rate = mutation_rate
#         self.pop_size = pop_size
#         self.pop = np.vstack([np.random.permutation(DNA_size) + np.random.choice([0, 1, 2, 3], DNA_size, replace=True) /
#                               10 for _ in range(pop_size*2)])

#     def select(self, fitness):
#         idx = np.random.choice(np.arange(self.pop_size*2), size=self.pop_size, replace=True, p=fitness / fitness.sum())
#         self.pop_p = self.pop[idx]
#         return self.pop_p,idx

#     def crossover(self, parent, pop):
#         if np.random.rand() < self.cross_rate:
#             i_ = np.random.randint(0, self.pop_size, size=1)     # select another individual from pop
#             ch_point =  np.random.randint(0, 2, self.DNA_size) 
#             change = np.where(ch_point==1)[0].tolist()  # 变了的位置索引
#             cross_points = ch_point.astype(np.bool)   # choose crossover points
#             keep_city = parent[~cross_points]   # [0.  2.1 3. ]  find the city number
#             keep_city_n = [int(i) for i in keep_city]  # [0, 2, 3]
#             parent_i = pop[i_].ravel()
#             parent_i_n = [int(i) for i in parent_i]  # [0, 2, 3]
#             # 但是当参数invert被设置为True时，情况恰好相反，如果parent_i_n中元素在keep_city_n中没有出现则返回True,如果出现了则返回False.
#             swap_city = pop[i_, np.isin(parent_i_n, keep_city_n, invert=True)]  # ravel将数组维度拉成一维数组
#             keep_city_list=keep_city.tolist()
#             for i,j in zip(change,swap_city):
#                 keep_city_list.insert(i,j)
#             new_pop = np.concatenate((keep_city, swap_city))
#             # print(new_pop,swap_city)
#             # print(keep_city_list)
#             # parent[:] = new_pop
#             parent[:] = np.array(keep_city_list)
#         return parent

#     def mutate(self, child):
#         for point in range(self.DNA_size):
#             if np.random.rand() < self.mutate_rate:
#                 swap_point = np.random.randint(0, self.DNA_size)
#                 swapA, swapB = child[point], child[swap_point]
#                 child[point] = math.modf(swapB)[1] + np.random.choice([0, 1, 2, 3]) / 10
#                 child[swap_point] = math.modf(swapA)[1] + np.random.choice([0, 1, 2, 3]) / 10
#         return child

#     def evolve(self, fitness):
#         pop,idx = self.select(fitness)
#         pop_copy = pop.copy()
#         for parent in pop:  # for every parent
#             child = self.crossover(parent, pop_copy)
#             child = self.mutate(child)
#             parent[:] = child
#         self.pop[idx] = pop

#     # 根据基因的编码数据获得总距离.gene_all表示种群基因
#     # area存储的是区域的四个可行起始点，行号代表区域号，列号代表方向号，共有四列，每一列是一个序列[a,b]
#     # (5, 4, 2, 2) 5行代表有五个城市，4代表每行有四个元素，2表示每个元素包含两个点，2代表每个点有(x,y)坐标
#     def get_dis(self, gene_all, start, area, yaw_task, curvature):
#         # gene_all = np.array([[3.3, 4. ,5. ,2.2,1.3,0.2]])
#         # print(gene_all,np.shape(gene_all))   (500, 6)
#         num_pop = gene_all.shape[0]
#         gene_len = gene_all.shape[1]

#         total_distance = []
#         point_all = []

#         city = [0, 1]
#         city_ = [0, 1]

#         # 遍历每个个体，计算其适应度
#         for j in range(num_pop):
#             gene = gene_all[j]
#             dis_all0 = 0
#             point = []
#             point.append(start)
#             # 遍历基因序列，计算相邻两个元素之间的距离
#             city[0] = int(math.modf(gene[0])[1])
#             city[1] = int(np.round(math.modf(gene[0])[0] * 10))
#             point0 = area[city[0]][city[1]][0]
#             point1 = area[city[0]][city[1]][1]

#             # yaw0 = yaw_task[city[0]][city[1]][0]
#             # yaw1 = yaw_task[city[0]][city[1]][1]
#             for i in range(gene_len-1):
#                 # 起始城市及其起点和终点
#                 point.append(point0)
#                 point.append(point1)
#                 # 目标城市及其起点和终点
#                 city_[1] = int(np.round(math.modf(gene[i+1])[0] * 10))
#                 city_[0] = int(math.modf(gene[i+1])[1])
#                 point0_ = area[city_[0]][city_[1]][0]
#                 point1_ = area[city_[0]][city_[1]][1]

#                 # yaw0_ = yaw_task[city_[0]][city_[1]][0]
#                 # yaw1_ = yaw_task[city_[0]][city_[1]][1]
#                 # 如果是第一个城市，则应该加上起点到这里的距离
#                 if i == 0:
#                     dis = np.linalg.norm(point0 - start)
#                     # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(start[0], start[1], 0, point0[0],
#                     #                                                                    point0[1], yaw0, curvature)
#                     dis_all0 = dis + dis_all0
#                     # print(dis)
#                     # dis_all1 = dis_all1+path_length
#                 # 如果是最后一个城市，则应该加上这里到起点的距离
#                 if i == len(gene)-2:
#                     point.append(point0_)
#                     point.append(point1_)
#                     dis = np.linalg.norm(point1_ - start)
#                     # print(dis)
#                     # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(point1_[0], point1_[1], yaw1_,
#                     #                                                                    start[0], start[1], pi, curvature)
#                     dis_all0 = dis + dis_all0
#                     # dis_all1 = dis_all1 + path_length
#                 # 获得距离计算的起点,上个城市的终点point1，下个城市的起点point0_
#                 dis = np.linalg.norm(point0_-point1)
#                 # print(dis)
#                 # 计算距离，真正的距离应该是杜宾斯规划出来的
#                 # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(point1[0], point1[1], yaw1,
#                 #                                                                    point0_[0], point0_[1], yaw0_, curvature)
#                 dis_all0 = dis + dis_all0
#                 # dis_all1 = dis_all1 + path_length
#                 # 其实城市位置更新
#                 point0 = point0_
#                 point1 = point1_
#                 # yaw0 = yaw0_
#                 # yaw1 = yaw1_
#             total_distance.append(dis_all0)
#             point_all.append(point)
#         total_distance = np.array(total_distance)
#         fitness = np.exp(self.DNA_size * 200 / total_distance)
#         return total_distance, fitness, point_all


# class TravelSalesPerson(object):
#     def __init__(self):
#         plt.ion()

#     def plotting(self, lx, ly, total_d, map, ax):
#         plt.cla()
#         map.map_plot(ax)
#         plt.scatter(lx, ly, s=100, c='k')
#         ax.plot(lx, ly, 'ro-')
#         plt.text(50, 50, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.01)


# def ga_order(static_raw, yaw_task=None, curvature=None):
#     # print(static,np.shape(static)) # torch.Size([8, 25])
#     area_num = int((np.shape(static_raw)[1]-1)/4)
#     static= static_raw.clone()
#     start = static[0:2,0]
#     area = get_ga_in(area_num,static)
#     # 获取任务目标点
#     N_CITIES = np.shape(area)[0]
#     cut = 1000
#     # 参数分别是城市数量，基因交叉率，基因突变率，种群数量
#     ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
#     # env = TravelSalesPerson()
#     count = 0
#     start_pos = start[0:2]
#     total_distance, fitness, point_all = ga.get_dis(ga.pop, start_pos, area, yaw_task, curvature)
#     while count < N_GENERATIONS:  # for generation in range(N_GENERATIONS):#N_GENERATIONS
#         ga.evolve(fitness)
#         total_distance, fitness, point_all = ga.get_dis(ga.pop, start_pos, area, yaw_task, curvature)
#         best_idx = np.argmax(fitness)
#         if count > 200:
#             print('Gen:', count, '| best dis: %.2f' % total_distance[best_idx], '| best fit: %.2f' % fitness[best_idx],)
#         new_cut1 = total_distance[best_idx]
#         # print(new_cut1)
#         # 如果最优适应性增加，则从头开始，一直到500个回合都不在增加，种群稳定才退出循环
#         if cut > new_cut1:
#             cut = new_cut1
#             count = 0
#         else:
#             count += 1
#         point_all = np.array(point_all)
#     # print("111 GA result:", ga.pop[best_idx])
#     # total_distance0, fitness0, point_0 = ga.get_dis(ga.pop[best_idx].reshape(1,-1), start_pos, area, yaw_task, curvature)
#     # print(ga.pop[best_idx].reshape(1,-1),total_distance0)
#     return ga.pop[best_idx], total_distance[best_idx] 


import matplotlib.pyplot as plt
import numpy as np
import math
import xlwt
CROSS_RATE = 0.1
MUTATE_RATE = 0.02
POP_SIZE = 500
N_GENERATIONS = 200
pi = 3.1415926
def get_yaw(end_pos,star_pos):
    theta = np.arctan((end_pos[1]-star_pos[1])/(end_pos[0]-star_pos[0])) if star_pos[0] != end_pos[0] else -pi/2# -pi
    if end_pos[0] < star_pos[0]:  # 在二三象限
        theta = theta+pi
    end_yaw = theta
    return end_yaw

def get_ga_in(area_num,static):
    # static:torch.Size([batch, 8, area_num*4+1])-torch.Size([8, area_num*4+1])
    area_task = [[] for _ in range(area_num)]
    for i in range(0, area_num): 
        for j in range(4):
            area_task[i].append([static[0:2,i*4+j+1], static[2:4,i*4+j+1]])
    return area_task


def choose_path(area_copy, start, mask):
    #(6, 4, 2)print(np.shape(area_copy))
    # 根据无人机当前的位置选择一个最近的航道
    length_all = []
    m = float('inf')
    for j in range(len(area_copy)):
        if j in mask:
            for i in range(4):
                length_all.append(m)
            continue
        area_j = area_copy[j]   #第j个区域（4，2，2）
        for i in range(4):
            area_j_i = area_j[i] # 第j个区域的第i个方向（2，2）
            entry = area_j_i[0]
            path_length = np.linalg.norm(entry.numpy()-np.array(start))
            length_all.append(path_length)
    index_min = np.argmin(length_all)
    # print(index_min)
    dis = length_all[index_min]
    area_index = int(index_min / 4) # 获得最近的区域
    path_index = index_min % 4 # 获得最近的方向
    path_num = area_index+path_index/10  # 整数部分表示区域，小数部分表示方向，用于遗传编码
    return index_min,dis,area_index,path_index


def greedy(static_raw):# torch.Size([8, 25])
    static = static_raw.clone()
    area_num = int((np.shape(static)[1]-1)/4)
    start = static[0:2,0]
    area = get_ga_in(area_num,static) # (area_num,4,2,2)
    start0 = (start.numpy()).copy()  
    start_position = (start.numpy()).copy()
    area_copy = area.copy()
    dis_length = 0
    greedy_order = []
    mask = [10]
    for i in range(0, area_num):
        # 根据当前位置选择一个最近的航道,绘制到达区域的路径，area_copy会被改变，删去已经选择的区域
        index_min, dis, area_index, path_index = choose_path(area_copy, start_position,mask)
        greedy_order.append(index_min+1)
        # print(index_min)
        # print("area_index:"+str(area_index))
        mask.append(area_index)
        # print("mask:"+str(mask))
        # 计算距离
        dis_length = dis_length + dis
        # 搜索完后，无人机的位置需要更新
        leave = area_copy[area_index][path_index][1]         #[2,2]
        start_position = [leave[0], leave[1]]
    # 返回原点
    start_position = [i.numpy()for i in start_position]
    start_position = np.array(start_position)
    path_length = np.linalg.norm(start_position-start0)
    dis_length = dis_length + path_length
    return greedy_order,dis_length



class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        self.pop = np.vstack([np.random.permutation(DNA_size) + np.random.choice([0, 1, 2, 3], DNA_size, replace=True) /
                              10 for _ in range(pop_size)])

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # choose crossover points
            keep_city = parent[~cross_points]   # [0.  2.1 3. ]  find the city number
            keep_city_n = [int(i) for i in keep_city]  # [0, 2, 3]
            parent_i = pop[i_].ravel()
            parent_i_n = [int(i) for i in parent_i]  # [0, 2, 3]
            swap_city = pop[i_, np.isin(parent_i_n, keep_city_n, invert=True)]  # ravel将数组维度拉成一维数组
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point] = math.modf(swapB)[1] + np.random.choice([0, 1, 2, 3]) / 10
                child[swap_point] = math.modf(swapA)[1] + np.random.choice([0, 1, 2, 3]) / 10
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop

    # 根据基因的编码数据获得总距离.gene_all表示种群基因
    # area存储的是区域的四个可行起始点，行号代表区域号，列号代表方向号，共有四列，每一列是一个序列[a,b]
    # (5, 4, 2, 2) 5行代表有五个城市，4代表每行有四个元素，2表示每个元素包含两个点，2代表每个点有(x,y)坐标
    def get_dis(self, gene_all, start, area, yaw_task, curvature):
        # gene_all = np.array([[3.3, 4. ,5. ,2.2,1.3,0.2]])
        # print(gene_all,np.shape(gene_all))   (500, 6)
        num_pop = gene_all.shape[0]
        gene_len = gene_all.shape[1]

        total_distance = []
        point_all = []

        city = [0, 1]
        city_ = [0, 1]

        # 遍历每个个体，计算其适应度
        for j in range(num_pop):
            gene = gene_all[j]
            dis_all0 = 0
            point = []
            point.append(start)
            # 遍历基因序列，计算相邻两个元素之间的距离
            city[0] = int(math.modf(gene[0])[1])
            city[1] = int(np.round(math.modf(gene[0])[0] * 10))
            point0 = area[city[0]][city[1]][0]
            point1 = area[city[0]][city[1]][1]

            # yaw0 = yaw_task[city[0]][city[1]][0]
            # yaw1 = yaw_task[city[0]][city[1]][1]
            for i in range(gene_len-1):
                # 起始城市及其起点和终点
                point.append(point0)
                point.append(point1)
                # 目标城市及其起点和终点
                city_[1] = int(np.round(math.modf(gene[i+1])[0] * 10))
                city_[0] = int(math.modf(gene[i+1])[1])
                point0_ = area[city_[0]][city_[1]][0]
                point1_ = area[city_[0]][city_[1]][1]

                # yaw0_ = yaw_task[city_[0]][city_[1]][0]
                # yaw1_ = yaw_task[city_[0]][city_[1]][1]
                # 如果是第一个城市，则应该加上起点到这里的距离
                if i == 0:
                    dis = np.linalg.norm(point0 - start)
                    # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(start[0], start[1], 0, point0[0],
                    #                                                                    point0[1], yaw0, curvature)
                    dis_all0 = dis + dis_all0
                    # print(dis)
                    # dis_all1 = dis_all1+path_length
                # 如果是最后一个城市，则应该加上这里到起点的距离
                if i == len(gene)-2:
                    point.append(point0_)
                    point.append(point1_)
                    dis = np.linalg.norm(point1_ - start)
                    # print(dis)
                    # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(point1_[0], point1_[1], yaw1_,
                    #                                                                    start[0], start[1], pi, curvature)
                    dis_all0 = dis + dis_all0
                    # dis_all1 = dis_all1 + path_length
                # 获得距离计算的起点,上个城市的终点point1，下个城市的起点point0_
                dis = np.linalg.norm(point0_-point1)
                # print(dis)
                # 计算距离，真正的距离应该是杜宾斯规划出来的
                # path_x0, path_y0, path_yaw0, _, path_length = dubins_path_planning(point1[0], point1[1], yaw1,
                #                                                                    point0_[0], point0_[1], yaw0_, curvature)
                dis_all0 = dis + dis_all0
                # dis_all1 = dis_all1 + path_length
                # 其实城市位置更新
                point0 = point0_
                point1 = point1_
                # yaw0 = yaw0_
                # yaw1 = yaw1_
            total_distance.append(dis_all0)
            point_all.append(point)
        total_distance = np.array(total_distance)
        fitness = np.exp(self.DNA_size * 200 / total_distance)
        return total_distance, fitness, point_all

best_dis = xlwt.Workbook()
sheet1 = best_dis.add_sheet(u'min', cell_overwrite_ok=True)
sheet2 = best_dis.add_sheet(u'ave', cell_overwrite_ok=True)
sheet3 = best_dis.add_sheet(u'cor', cell_overwrite_ok=True)
class TravelSalesPerson(object):
    def __init__(self):
        plt.ion()

    def plotting(self, lx, ly, total_d, map, ax):
        plt.cla()
        map.map_plot(ax)
        plt.scatter(lx, ly, s=100, c='k')
        ax.plot(lx, ly, 'ro-')
        plt.text(50, 50, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.01)


def ga_order(static_raw, yaw_task=None, curvature=None):
    # print(static,np.shape(static)) # torch.Size([8, 25])
    area_num = int((np.shape(static_raw)[1]-1)/4)
    static= static_raw.clone()
    start = static[0:2,0]
    area = get_ga_in(area_num,static)
    # 获取任务目标点
    N_CITIES = np.shape(area)[0]
    cut = 1000
    # 参数分别是城市数量，基因交叉率，基因突变率，种群数量
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
    # env = TravelSalesPerson()
    count = 0
    start_pos = start[0:2]
    step = 0 
    total_distance, fitness, point_all = ga.get_dis(ga.pop, start_pos, area, yaw_task, curvature)
    while count < N_GENERATIONS:  # for generation in range(N_GENERATIONS):#N_GENERATIONS
        step = step + 1
        ga.evolve(fitness)
        total_distance, fitness, point_all = ga.get_dis(ga.pop, start_pos, area, yaw_task, curvature)
        best_idx = np.argmax(fitness)
        if count > 200:
            print('Gen:', count, '| best dis: %.2f' % total_distance[best_idx], '| best fit: %.2f' % fitness[best_idx],)
        new_cut1 = total_distance[best_idx]
        m = np.mean(total_distance) # 均值
        n = np.std(total_distance) #方差
        min_m = np.min(total_distance) #方差

        sheet1.write(step-1,0,m)
        sheet1.write(step-1,1,n)
        sheet1.write(step-1,2,min_m)
        # print(new_cut1)
        # 如果最优适应性增加，则从头开始，一直到500个回合都不在增加，种群稳定才退出循环
        if cut != new_cut1:
            cut = new_cut1
            count = 0
        else:
            count += 1
        point_all = np.array(point_all)
    best_dis.save("./best_dis.xls")
    # print("111 GA result:", ga.pop[best_idx])
    # total_distance0, fitness0, point_0 = ga.get_dis(ga.pop[best_idx].reshape(1,-1), start_pos, area, yaw_task, curvature)
    # print(ga.pop[best_idx].reshape(1,-1),total_distance0)
    return ga.pop[best_idx], total_distance[best_idx],step 






