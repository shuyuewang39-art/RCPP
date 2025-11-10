import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v = torch.FloatTensor(n_hidden,4).cuda() #torch.Size([128, 4])
        # v1 = torch.FloatTensor(n_hidden) # torch.Size([128])
        self.v = nn.Parameter(v) # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden)) # torch.Size([128, 4])

        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)

    def forward(self, q, ref):    # query and reference
        # ref--(B *size, dim)
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)

        # 对 query 进行变换
        q = self.Wq(q)     # (B, dim)
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size+1, dim)

        # 对 ref 进行变换    
        ref = self.Wref(ref) # (B, dim)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size+1, dim)
        
        # v_view: (B, dim, 1), 归一化向量
        # (dim,4)-(1,dim,4)-(B,dim,4)-(B,dim,1)修改版本
        # (dim,)-(1,dim)-(B,dim)-(B,dim,1)原始版本
        # self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim, 4)

        # (B, size, dim) * (B, dim, 4)-(B, (size+1)*4)
        # u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).reshape(self.batch_size, self.size*4)
        u = u[:,3:]

        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)

    def forward(self, x, h, c):       # query and reference
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        h = o * torch.tanh(c)
        return h, c


class GPN(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        # n_feature=8, n_hidden=128
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(int(n_feature/2), n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)

        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        # x=[batch_size,8],X_all=[batch_size,city_size,8]
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)

        # vector contex  trepeat是先复制原有的，再拼接
        x_expand = x.unsqueeze(1).repeat(1, self.city_size, 1)   # (batch_size,city_size,4)
        x_ex = x_expand[:,:,2:4].repeat(1,1,4)
        X_all = X_all-x_ex #所有点减去当前选择的区域的离开点 
        
        # 嵌入层
        x = self.embedding_x(x) #  (batch_size,city_size,h)
        context = self.embedding_all(X_all)# (batch_size,city_size,h)
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1, 0, 2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)

        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.view(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context))

        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        
        
        mask = np.repeat(mask.cpu(), 4, axis=1)
        mask = mask.cuda()
        u = 10 * torch.tanh(u) + mask[:,3:]
        
        if latent is not None:
            u += self.alpha * latent
    
        return F.softmax(u, dim=1), h, c, latent_u
