import numpy as np

# a=np.arange(6)
# b=np.insert(a, 3, -1)
# idxList = list(range(len(b)))
# zeroIdx = np.asarray(idxList)[b == -1]  # 将结构数据转化为ndarray
# print(a, b, zeroIdx)



order0 = [-1, 2, -1, 1, 5, 3, 0, 6, 4, -1]
order1=[0, 2, 0, 1, 3, 2, 2, 1, 2, 0]
def num_e(l,target):
    b = []
    for index, nums in enumerate(l):
        if nums == target:
            b.append(index)
    return b
order_i = []
order_j = []
index = num_e(order0, -1)  # order0中-1的索引
# print(index)
for i, j in zip(index[0::], index[1::]):
    o1 = order0[i+1:j]
    o2 = order1[i+1:j]
    order_i.append(o1)
    order_j.append(o2)
# print(order_i)
# print(order_j)