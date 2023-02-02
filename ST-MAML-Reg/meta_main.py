import torch
from syn_data_loader import ManyFunctionsMetaDataset
from meta_train import meta_train_one_epoch, meta_eval_one_epoch
from meta_model import EncModel
import torch.optim as optim
import os
import argparse
import numpy as np
import glob
from utils import get_saved_file


def get_arguments():
    parser = argparse.ArgumentParser(description='ST-MAML')
    parser.add_argument('--num_val', type=int, default=10)
    parser.add_argument('--num_sample_function', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--meta_bs', type=int, default=36)
    parser.add_argument('--epoch', type=int, default=510)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--total_batches',  type=int, default=500)
    parser.add_argument('--inner_loop_grad_clip', type=float, default=10)
    parser.add_argument('--noise_std', default=0.3, type=float)
    parser.add_argument('--resume_train', default=False, action='store_true')
    parser.add_argument('--inner_lr', default=1e-3, type=float)
    parser.add_argument('--inner_step', default=3, type=int)
    parser.add_argument('--img_size', default=28, type=int)
    parser.add_argument('--loss_type', default='MSEloss', choices=['BCEloss', 'MSEloss'])
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--kl_weight', default=2., type=float)
    parser.add_argument('--in_weight_rest', default=0.001, type=float)
    parser.add_argument('--aug_enc', default=True, action='store_true')
    parser.add_argument('--act_type', default='softplus', choices=['softplus', 'sigmoid'])
    parser.add_argument('--model_type', default='prob', choices=['prob', 'deter'])
    return parser.parse_args()

args = get_arguments()


dataset_param = {'noise_std': args.noise_std, 'num_total_batches': args.total_batches, 'num_samples_per_function':args.num_sample_function, \
                 'num_val_samples': args.num_val, 'meta_batch_size':args.meta_bs}
dataset = ManyFunctionsMetaDataset(**dataset_param)



if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu_id)
else:
    device = 'cpu'
print(device)
print(args.resume_train)
state_str = os.path.join(args.output_folder, args.model_type+'_'+args.loss_type+'_'+str(args.inner_step)+\
    '_'+str(args.kl_weight)+'_'+str(args.in_weight_rest)+'_'+str(args.aug_enc)+'_'+str(args.act_type))

args.save_epoch_loss = os.path.join(state_str, 'save_epoch_loss.txt')
if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)
if not os.path.exists(state_str):
    os.mkdir(state_str)

    
if not os.path.exists(args.save_epoch_loss):
    f = open(args.save_epoch_loss, 'a+')
    
    



model = EncModel(2, [20, 40], 1, 20, [80, 80], 40, 3, [20, 20],20, aug_enc=args.aug_enc,model_type=args.model_type, act_type=args.act_type)
model.to(device)


print(model)

if args.resume_train:
    path = state_str
    start_epoch = get_saved_file(path)
    start_model_str = os.path.join(state_str, 'epoch_'+str(start_epoch)+'.pt')
    model.load_state_dict(torch.load(start_model_str))
    start_epoch += 1

else:
    start_epoch=0
    



opt = optim.Adam(model.parameters(), lr=args.lr)



eval_loss_init = 100
for epoch in range(start_epoch, args.epoch):
    meta_train_one_epoch(model, dataset, opt, epoch, args, 'train', device)
    eval_loss = meta_eval_one_epoch(model, dataset, opt, epoch, args, 'eval', device)
    if eval_loss<eval_loss_init:
        eval_loss_init = eval_loss
        torch.save(model.state_dict(), os.path.join(state_str, 'best.pt'))

    if (epoch+1)%10==0:
        torch.save(model.state_dict(), os.path.join(state_str, 'epoch_'+str(epoch)+'.pt'))
        
        
        

