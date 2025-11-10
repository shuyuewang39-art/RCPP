import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder, Pointer, Attention
import numpy as np
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
class DRL4TSP(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, static_city, dis_all, max_load, decoder_input=None, last_hh=None):
        # 在训练前，初始化一些参数
        # static:dtype=torch.float64,torch.Size([256, 8, size+1])
        # static_city = static_city.transpose(1, 2)
        # static_city, dis_all=static_city.cuda(),dis_all.cuda()
        # static_city, dis_all=static_city.cuda(),dis_all.cuda()
        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1) # torch.Size([256, 2, 1])

        # torch.Size([256, size+1])
        mask = torch.ones(batch_size, sequence_size, device=device) 
        mask =  torch.repeat_interleave(mask, 4, dim=1)[:,3:]
        # 存储上一个城市的索引
        last_idx = [0]*batch_size  # 用于计算负载的消耗，第一个城市是原点
        last_idx = torch.tensor(last_idx,dtype=torch.int64) 
        last_idx = last_idx.unsqueeze(1)#.cuda() # torch.Size([256，1])

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # 扩维操作
        static_hidden = self.static_encoder(static) # [b,h,s+1]
        dynamic_hidden = self.dynamic_encoder(dynamic) # [b,h,s+1]
        for _ in range(max_steps):
            if not mask.byte().any(): # mask中全为假，这个条件才为真，这时跳出循环
                break

            decoder_hidden = self.decoder(decoder_input) # [b,h,s+1]
            # 输出的是选择几个节点的概率  [b,s*4]
            probs, last_hh = self.pointer(static_hidden,
                                          dynamic_hidden,
                                          decoder_hidden, last_hh)
            # print(torch.min(probs))
            # mask_city =  torch.repeat_interleave(mask, 4, dim=1)[:,3:] # 每一个元素重复四次放在自己身旁[b,4（s+1）]
            # a=probs + mask.log()
            # print(a)
            probs = F.softmax(probs + mask.log(), dim=1)   # torch.Size([256, 11])
            
            # print(torch.min(probs))
            # print("---------------------")
            # ptr表示选择城市的索引，logp表示选择城市的概率
            if self.training:
                # 按照probs的概率，在相应的位置进行采样，采样返回的是该位置的整数索引。
                m = torch.distributions.Categorical(probs) # m是Categorical(probs: torch.Size([256, 11]))
                ptr = m.sample() # torch.Size([256])对256个样本抽样的结果

                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr) # torch.Size([256])
            else:
                # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            # 根据网络的输出选择更新负载和需求
            if self.update_fn is not None:
                # [b,2,s+1],[b],[b,1]
                dynamic = self.update_fn(dynamic, ptr.data, last_idx,static_city,max_load)
                dynamic_hidden = self.dynamic_encoder(dynamic)
                # 如果存在样本中的需求总数等于0，则说明提前结束了.提前结束的就将选择所有城市的概率变为0
                is_done = dynamic[:, 1].sum(1).eq(0).float() #dynamic[:, 1]表示需求，(batch_size, num)
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data, dis_all,static_city).detach()
                
            # 把每次选择按照顺率记录下来
            last_idx =  ptr.data.unsqueeze(1)  
            tour_logp.append(logp.unsqueeze(1)) # torch.Size([256，1])
            tour_idx.append(ptr.data.unsqueeze(1))  # torch.Size([256，1])
            # torch.Size([256，1，1])-torch.Size([256，2，1])# torch.Size([256, 2, 1])
            # static_city = static_city.transpose(1, 2)
            decoder_input = torch.gather(static_city, 2,ptr.view(-1, 1, 1).expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, 16)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp