from posixpath import split
import torch
import os
import backbone
import numpy as np
from st_maml import ST_MAML
from weather_loader import NOAA_GSOD_MetaDset
from maml import MAML
from io_utils import model_dict, get_arguments, get_resume_file, get_best_file
import random
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader



def train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), params.lr)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    max_acc = 100000       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc, _ = model.test_loop( val_loader)
        if acc < max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model



if __name__=='__main__':

    params = get_arguments()


    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(params.gpu_id)
    else:
        device = 'cpu'
    print(device)


    
    base_data = NOAA_GSOD_MetaDset(split='train')
    base_loader = DataLoader(base_data, batch_size=params.meta_bs, shuffle=True)
    val_data = NOAA_GSOD_MetaDset(split='val')
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)


    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    if params.method =='ST_MAML':
        model = ST_MAML(model_dict[params.model], **train_few_shot_params)
    elif params.method == 'MAML':
        model = MAML(model_dict[params.model], **train_few_shot_params)

    model = model.to(device)

    model.n_task     = params.meta_bs
    model.task_update_num = params.inner_step
    model.train_lr = params.inner_lr
    model.in_weight_rest = params.in_weight_rest
    model.kl_weight = params.kl_weight


    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    


    params.checkpoint_dir = '%scheckpoints/%s/%s_%s_%s_%s_%s_%s_%s_%s_%s' %(params.output_folder, 'Weather', params.method, params.model, str(params.inner_step), \
        str(params.embed_size), str(params.aug_fraction),str(params.inner_lr), str(params.in_weight_rest), str(params.meta_bs), str(params.kl_weight))
    


    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(model)

    if params.resume_train:
        resume_file = get_best_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
        print('***********reload model successfully***********')
    

   

    optimization='Adam'

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)



    
