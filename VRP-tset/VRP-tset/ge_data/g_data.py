import numpy as np
import os
import random
'''self.widthxself.width的地图,平均分成self.max_col*self.max_col个小正方形,每个正方形self.unitxself.unit;
从这些小正方形中随机抽取self.area_num个用来生成覆盖区域'''
# 生成数据集
class get_data():
    def __init__(self, samples, area_num,seed=None):
        self.pi=3.1415
        self.sample=samples                            # 样本个数
        self.area_num=area_num                         # 多少个子区域
        self.seed = seed
        self.width=160
        self.max_col=4 # 4:25,5:20
        self.unit=int(self.width/self.max_col) #单位刻度的长度
        self.sub_grid=[0,1,2,3]                   # 子区域行号索引
    
    def choose_grid(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        # 各个子区域所在的子栅格索引：第几行,第几列
        x_start,y_start=[],[]
        x_index,y_index=[],[]
        for mm in range(self.sample):
            indedx = np.array(random.sample(range(self.max_col*self.max_col), self.area_num+1)) # 从25个区域中选择一个
            i = indedx // self.max_col
            j = indedx % self.max_col
            x_index.append(i[1:])
            y_index.append(j[1:])
            x_start.append(i[0])
            y_start.append(j[0])
        x_start,y_start=np.array(x_start),np.array(y_start)
        xs_min, xs_max=x_start*self.unit, (x_start+1)*self.unit  
        ys_min, ys_max=y_start*self.unit, (y_start+1)*self.unit  
        self.startx = np.random.randint(xs_min+5, xs_max-5,size=(self.sample,1))
        self.starty = np.random.randint(ys_min+5, ys_max-5,size=(self.sample,1))
        self.start = np.hstack((self.startx,self.starty)).reshape(self.sample,1,2)

        x_index, y_index = np.array(x_index), np.array(y_index)# (10, 5)
        # 确定每个栅格区域的坐标范围
        x_min, x_max=x_index*self.unit, (x_index+1)*self.unit  # (10, 5)
        y_min, y_max=y_index*self.unit, (y_index+1)*self.unit  # ()
        # 将J列当成一个整体重复4遍，这样可以一次性生成四个符合约定的横坐标
        X0=np.repeat(x_min.copy(),4,axis=1) #(10, 20)
        X1=np.repeat(x_max.copy(),4,axis=1) #(10, 20)
        # X0=np.repeat(x_min.copy(),4,axis=1) #(10, 20)
        # X1=np.repeat(x_max.copy(),4,axis=1) #(10, 20)
        X02=np.repeat(x_min.copy(),2,axis=1)  #(10, 10)
        X12=np.repeat(x_max.copy(),2,axis=1)  #(10, 10)
        Y0=np.repeat(y_min.copy(),2,axis=1)  #(10, 10)
        Y1=np.repeat(y_max.copy(),2,axis=1)  #(10, 10)
        return X0,X1,Y0,Y1,y_min, y_max, x_min, x_max

    '''每个子栅格区域,生成子区域:子区域应该是由两条平行线构成的梯形'''
    '''在每个子栅格生成两个固定的点,以及一个角,
    根据这个斜率和两个点生成两条射线，再生成另外两个点的纵坐标'''
    def ge_area3(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        X0, X1, Y0, Y1, y_min, y_max, x_min, x_max = self.choose_grid()
        x_num, y_num = self.area_num*4, self.area_num*2
        # x_4中的顺序是，x0,x2,x1,x3,前4个是第一个区域的点
        x_4 = np.random.uniform(X0+5, X1-5, size=(self.sample, self.area_num*4))
        # 对x0和x1控制x1距离x0远一些，否则，斜率的计算可能不准确；
        delt1, delt2 = x_4[:,2:x_num:4] - x_4[:,0:x_num:4], x_4[:,1:x_num:4] - x_4[:,0:x_num:4]  # x1-x0
        falg_check1, falg_check2 =(abs(delt1)<5).astype(int),(abs(delt2)<5).astype(int) # 与第一个点距离是否太近,太近就是1
        xn_1 = x_4[:,0:x_num:4] + np.sign(delt1)*5+falg_check1*delt1
        xn_2 = x_4[:,0:x_num:4] + np.sign(delt2)*15+falg_check2*delt2
        xn_1,xn_2 = xn_1.clip(x_min,x_max), xn_2.clip(x_min,x_max)
        x_4[:,1:x_num:4], x_4[:,2:x_num:4] = xn_2, xn_1
        # y_0_2 中的顺序是，y0,y2, 前2个是第一个区域的点
        y_0_2 = np.random.uniform(Y0+5, Y1-5, size=(self.sample, self.area_num*2))
        # 斜率,tan(85°)=11.43，tan(5°)=0.088
        k = np.random.choice((-1,1), size=(self.sample, self.area_num))*np.random.uniform(0.088, 11.43, size=(self.sample, self.area_num))
        # 获得x1,x3的取值范围
        x1_left = (y_max - y_0_2[:, 0:y_num:2])/k + x_4[:, 0:x_num:4]
        x1_right = (y_min - y_0_2[:, 0:y_num:2])/k + x_4[:, 0:x_num:4]
        x1_min, x1_max=np.minimum(x1_left,x1_right),np.maximum(x1_left,x1_right)
        x1_min, x1_max = x1_min.clip(x_min,x_max), x1_max.clip(x_min,x_max)
        x_4[:,2:x_num:4]=x_4[:,2:x_num:4].clip(x1_min, x1_max)

        x3_left = (y_max - y_0_2[:, 1:y_num:2])/k + x_4[:, 1:x_num:4]
        x3_right = (y_min - y_0_2[:, 1:y_num:2])/k + x_4[:, 1:x_num:4]
        x3_min, x3_max=np.minimum(x3_left,x3_right),np.maximum(x3_left,x3_right)
        x3_min, x3_max = (x3_min.clip(x_min,x_max))+0, (x3_max.clip(x_min,x_max))-0
        x_4[:,3:x_num:4]=x_4[:,3:x_num:4].clip(x3_min, x3_max) # clip会自动取整
        #  获得y1,y3的取值范围
        y1 = k*(x_4[:, 2:x_num:4] - x_4[:, 0:x_num:4]) + y_0_2[:, 0:y_num:2] #(x1,y1)和(x0,y0)斜率一样
        y3 = k*(x_4[:, 3:x_num:4] - x_4[:, 1:x_num:4]) + y_0_2[:, 1:y_num:2] #(x3,y3)和(x2,y2)斜率一样
        # 获得y值
        y_4 = np.hstack((y_0_2,y1,y3)) # 0,2,1,3 (10, 20)
        # 获得y坐标，y0,y2,y1,y3,前4个是第一个区域的点
        y = np.zeros((self.sample,4))
        for ii in range(self.area_num):
            y_temp = np.hstack((y_0_2[:,ii*2:(ii+1)*2],y1[:,ii].reshape(self.sample,1),y3[:,ii].reshape(self.sample,1)))
            y=np.hstack((y,y_temp))
        y_4=y[:,4:]  # print(y_4)
        # 获得完整的位置特征
        pos = np.stack((x_4, y_4), axis=2) # (sample, city_num, 2)
        pos_all = np.concatenate((self.start,pos),axis = 1) # (sample, city_num+1, 2)
        return pos, pos_all

    def get_state3(self,X):
        sample=np.shape(X)[0]
        x0 = X[:,0,:] # 取出一个起始位置 # 0
        x0 = np.tile(x0,4)   #(2, 8)
        Y  = x0.reshape(sample,1,8)
        Y_n  = x0.reshape(sample,1,8)
        # print(X[0]) 
        for i in range(1,np.shape(X)[1],4): 
            x_area = X[:,i:i+4,:] # 取出一个(2, 4, 2)的数组
            x1=x_area.reshape(sample,1,8) #(2, 4*2)
            x_1 = x1.copy()
            # print(x_1,np.shape(x_1))

            x_area[:,[0,1],:]=x_area[:,[1,0],:]   # 1,0,2,3
            x2 = x_area.reshape(sample,1,8) 
            x_2 = x2.copy()
            # print(x_2,np.shape(x_2))
        
            x_area[:,[0,2],:]=x_area[:,[2,0],:]   # 2,0,1,3
            x_area[:,[1,3],:]=x_area[:,[3,1],:]   # 2,3,1,0
            x3 = x_area.reshape(sample,1,8) 
            x_3 = x3.copy()
            # print(x_3,np.shape(x_3))

            x_area[:,[0,1],:]=x_area[:,[1,0],:]   # 3,2,1,0
            x4 = x_area.reshape(sample,1,8) 
            x_4 = x4.copy()
            # print(x_4,np.shape(x_4))

            y_1 = np.concatenate((x_1,x_2,x_3,x_4),axis=1) # (2, 4, 8)一个区域的四个点
            # y1 = np.hstack((x_1,x_2,x_3,x_4)) # (2, 4, 8)一个区域的四个点

            Y = np.concatenate((Y,y_1),axis = 1)
            Y_n = np.concatenate((Y_n,x_1),axis = 1) # 只包含第一个x1，不含x2,x3,x4

        # print(Y,np.shape(Y)) #(sample, CITY_NUM+1, 8)
        Y_T = Y.transpose(0,2,1)
        return Y, Y_T, Y_n
    

# datas = get_data(1, 4, seed=1)  
# pos, pos_all = datas.ge_area3() # (sample, city_num, pos)
# Y, Y_T,Y_n = datas.get_state3(pos_all)
# print(Y)
# print(np.shape(Y)) #(sample, city_num+1, f) f=8
# print(Y_n)
# print(np.shape(Y_n)) #(sample, city_num/4+1, f) f=8

