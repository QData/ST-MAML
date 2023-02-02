from posixpath import split
import torch
import os
import backbone
import numpy as np
from st_maml import ST_MAML
from weather_loader import NOAA_GSOD_MetaDset
from io_utils import model_dict, get_arguments, get_resume_file, get_best_file
import random
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader


def eval(test_loader, model, params):    

        model.eval()
        acc = model.test_loop(test_loader)
        print(acc)



if __name__=='__main__':

    params = get_arguments()


    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(params.gpu_id)
    else:
        device = 'cpu'
    print(device)
    
    test_loader_param = {'split': 'test', 'num_classes': params.train_n_way, 'train_images_per_class':params.n_shot}
    
    test_data = NOAA_GSOD_MetaDset(split='test')
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)





    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    model = ST_MAML(model_dict[params.model], **train_few_shot_params)
   
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
    


    print(model)

    if params.resume_train:
        resume_file = get_best_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
            print(start_epoch)
            print('***********reload model successfully***********')
    
    model = eval(test_loader, model,  params)



    
