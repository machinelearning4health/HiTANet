import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time
class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(input_size, hidden_size*4)
        self.U_all = nn.Linear(hidden_size, hidden_size*4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional
        emb = nn.Embedding(input_size + 1, hidden_size, padding_idx=input_size)
        self.emb = emb

    def forward(self, inputs_list, reverse=False):
        # inputs: [b, seq, hid]
        # h: [b, hid]
        # c: [b, hid]
        inputs = inputs_list[0]
        timestamps = 1/torch.log(inputs_list[2]+2.7183)
        mask = inputs_list[1]
        inputs = (self.emb(inputs) * mask.unsqueeze(-1)).sum(dim=2)
        b,seq,hid = inputs.size()
        h = Variable(torch.Tensor(b,hid).zero_(), requires_grad=False)
        c = Variable(torch.randn(b,hid).zero_(), requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:,s:s+1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h)+self.U_all(inputs[:,s])
            f, i, o, c_tmp = torch.chunk(outs,4,1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.sigmoid(c_tmp)
            c = f*c_adj + i*c_tmp
            h = o*F.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs,1)
        return outputs