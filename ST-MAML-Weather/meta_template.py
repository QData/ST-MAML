import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.aug_dim = self.feature.aug_dim
        self.in_dim = self.feature.input_dim
        self.out_dim = self.feature.output_dim
        self.hid_dim = self.feature.hidden_dim

    @abstractmethod
    def set_forward(self,x,y, is_feature=False):
        pass

    @abstractmethod
    def set_forward_loss(self, x, y):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out


    

    def train_loop(self, epoch, train_loader, optimizer ):
        print_freq = 10

        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
            optimizer.zero_grad()
            loss = self.set_forward_loss( x )
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.data[0]

            if i % print_freq==0:
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))


