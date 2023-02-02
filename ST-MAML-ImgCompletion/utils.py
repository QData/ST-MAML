
import numpy as np
import torch
from torchvision import transforms
import glob
import os

def get_saved_file(path):
    filelist = glob.glob(path+'/*.pt')
    if len(filelist) == 0:
        return None
    else:
        pass
        
    index = np.array([int(os.path.splitext(os.path.basename(i).rsplit('_')[-1])[0]) for i in filelist])
    max_index = np.max(index)
    epoch = max_index
    
    return epoch








class MyCollator(object):
    def __init__(self, n_spt, n_val):
        self.n_spt = n_spt
        self.n_val = n_val
    def __call__(self, batch):
        assert isinstance(batch[0], tuple)

        batch_size = len(batch)

        num_total_points = self.n_val
        num_context = self.n_spt  # half of total points

        context_x, context_y, target_x, target_y = list(), list(), list(), list()

        for d, _ in batch:
            total_idx = range(784)
            total_idx = list(map(lambda x: (x // 28, x % 28), total_idx))
            c_idx = np.random.choice(range(784), num_total_points, replace=False)
            c_idx = list(map(lambda x: (x // 28, x % 28), c_idx))
            c_idx = c_idx[:num_context]
            c_x, c_y, total_x, total_y = list(), list(), list(), list()
            for idx in c_idx:
                c_y.append(d[:, idx[0], idx[1]])
                c_x.append((idx[0] / 27., idx[1] / 27.))
            for idx in total_idx:
                total_y.append(d[:, idx[0], idx[1]])
                total_x.append((idx[0] / 27., idx[1] / 27.))
            c_x, c_y, total_x, total_y = list(map(lambda x: torch.FloatTensor(x), (c_x, c_y, total_x, total_y)))
            c_y = c_y.unsqueeze(dim=1)
            total_y = total_y.unsqueeze(dim=1)
            context_x.append(c_x)
            context_y.append(c_y)
            target_x.append(total_x)
            target_y.append(total_y)


        return context_x, context_y, target_x, target_y












def collate_fn(batch, total_sample, n_spt):
    # Puts each data field into a tensor with outer dimension batch size
    assert isinstance(batch[0], tuple)

    batch_size = len(batch)

    num_total_points = total_sample
    num_context = n_spt  # half of total points

    context_x, context_y, target_x, target_y = list(), list(), list(), list()

    for d, _ in batch:
        total_idx = range(784)
        total_idx = list(map(lambda x: (x // 28, x % 28), total_idx))
        c_idx = np.random.choice(range(784), num_total_points, replace=False)
        c_idx = list(map(lambda x: (x // 28, x % 28), c_idx))
        c_idx = c_idx[:num_context]
        c_x, c_y, total_x, total_y = list(), list(), list(), list()
        for idx in c_idx:
            c_y.append(d[:, idx[0], idx[1]])
            c_x.append((idx[0] / 27., idx[1] / 27.))
        for idx in total_idx:
            total_y.append(d[:, idx[0], idx[1]])
            total_x.append((idx[0] / 27., idx[1] / 27.))
        c_x, c_y, total_x, total_y = list(map(lambda x: torch.FloatTensor(x), (c_x, c_y, total_x, total_y)))
        context_x.append(c_x)
        context_y.append(c_y)
        target_x.append(total_x)
        target_y.append(total_y)

    # context_x = torch.stack(context_x, dim=0)
    # context_y = torch.stack(context_y, dim=0).unsqueeze(-1)
    # target_x = torch.stack(target_x, dim=0)
    # target_y = torch.stack(target_y, dim=0).unsqueeze(-1)

    return context_x, context_y, target_x, target_y





