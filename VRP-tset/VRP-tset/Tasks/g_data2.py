import numpy as np


def get_state(X):
    Y=np.zeros((4,8))
    for i in range(0,np.shape(X)[0],4):
        x_area = X[i:i+4,:] # 取出一个四行四列的数组
        x1=x_area.flatten() # 0,1,2,3

        x_area[[0,1],:]=x_area[[1,0],:]   # 1,0,2,3
        x2=x_area.flatten()

        x_area[[0,2],:]=x_area[[2,0],:]   # 2,0,1,3
        x_area[[1,3],:]=x_area[[3,1],:]   # 2,3,1,0
        x3=x_area.flatten()

        x_area[[0,1],:]=x_area[[1,0],:]   # 3,2,1,0
        x4=x_area.flatten()

        Y1=np.vstack((x1,x2,x3,x4))
        Y=np.vstack((Y,Y1))
    state = Y[4:,:]
    # state = np.transpose(state)
    return state

