import torch
from syn_data_loader import *
import argparse
from meta_model import EncModel
import matplotlib.pyplot as plt
from syn_data_loader import ManyFunctionsMetaDataset
from meta_train import meta_train_one_epoch, meta_test_plot
from meta_model import EncModel
import torch.optim as optim
import os
import argparse
import numpy as np
import glob
from utils import get_saved_file

torch.manual_seed(0)
np.random.seed(0)

def get_arguments():
    parser = argparse.ArgumentParser(description='ST-MAML')
    parser.add_argument('--num_val', type=int, default=10)
    parser.add_argument('--num_sample_function', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--meta_bs', type=int, default=36)
    parser.add_argument('--epoch', type=int, default=510)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--total_batches',  type=int, default=2)
    parser.add_argument('--inner_loop_grad_clip', type=float, default=10)
    parser.add_argument('--noise_std', default=0.3, type=float)
    parser.add_argument('--resume_train', default=False, action='store_true')
    parser.add_argument('--inner_lr', default=1e-3, type=float)
    parser.add_argument('--inner_step', default=3, type=int)
    parser.add_argument('--img_size', default=28, type=int)
    parser.add_argument('--loss_type', default='MSEloss', choices=['BCEloss', 'MSEloss'])
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--kl_weight', default=2., type=float)
    parser.add_argument('--in_weight_rest', default=0.1, type=float)
    parser.add_argument('--aug_enc', default=True, action='store_true')
    parser.add_argument('--act_type', default='softplus', choices=['softplus', 'sigmoid'])
    parser.add_argument('--model_type', default='prob', choices=['prob', 'deter'])
    return parser.parse_args()

args = get_arguments()


dataset_param = {'noise_std': args.noise_std, 'num_total_batches': args.total_batches, 'num_samples_per_function':args.num_sample_function, \
                 'num_val_samples': args.num_val, 'meta_batch_size':args.meta_bs}
dataset = ManyFunctionsMetaDataset(**dataset_param)




device = 'cpu'
print(device)
print(args.resume_train)
state_str = os.path.join(args.output_folder, args.model_type+'_'+args.loss_type+'_'+str(args.inner_step)+\
    '_'+str(args.kl_weight)+'_'+str(args.in_weight_rest)+'_'+str(args.aug_enc)+'_'+str(args.act_type), 'best.pt')



model = EncModel(2, [20, 40], 1, 20, [80, 80], 40, 3, [20, 20],20, aug_enc=args.aug_enc,model_type=args.model_type, act_type=args.act_type)
model.to(device)


model.load_state_dict(torch.load(state_str, map_location=device))
model.eval()

out_list_plot = meta_test_plot(model, dataset, args, device=device)

#
#
def simple_visualize(out_list):
    bs = len(out_list)
    for i in range(15):

        j = np.random.choice(bs)
        train_set = out_list[j][0]
        if train_set[2]['task_id'] in [4, 5]:
            continue


        x_gt, y_gt, title = plot_gt(train_set)
        pred = out_list[j][1]
        spt_x = train_set.x[:, 0]
        spt_y = train_set[1].squeeze()
        qry_x = torch.linspace(-5, 5, steps=100)
        pred_qry_y = pred.squeeze().detach().numpy()
        plt.figure()
        plt.title(title)
        plt.plot(x_gt, y_gt)
        plt.scatter(spt_x, spt_y, c='r')
        plt.scatter(qry_x, pred_qry_y, c='b', s=0.9)

        plt.savefig('./plot/visual_{}.png'.format(i))




def plot_gt(task):
    task_id = task[2]['task_id']
    input_range = [-5, 5]
    x = np.linspace(*input_range, 1000)
    if task_id == 0:
        amp = task[2]['amp']
        phase = task[2]['phase']
        freq = task[2]['freq']
        y = amp * np.sin(freq * x - phase)
        title = 'sin curve'
    elif task_id == 1:
        slope = task[2]['slope']
        intersect = task[2]['intersect']
        y = x * slope + intersect
        title = 'straight line'
    elif task_id == 2:

        quad_coef = task[2]['quad_coef2']
        linear_coef = task[2]['linear_coef2']
        const_coef = task[2]['constant_coef2']
        y = quad_coef * x ** 2 + linear_coef * x + const_coef
        title = 'quadratic curve'

    elif task_id == 3:
        cubic_coef = task[2]['cubic_coef3']
        quad_coef = task[2]['quad_coef3']
        linear_coef = task[2]['linear_coef3']
        const_coef = task[2]['constant_coef3']

        y = cubic_coef * x ** 3 + quad_coef * x ** 2 + linear_coef * x + const_coef
        title = 'cubic curve'

    return x, y, title


#
simple_visualize(out_list_plot)