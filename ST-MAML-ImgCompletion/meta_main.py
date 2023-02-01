import torch
from meta_train import meta_eval_one_epoch, meta_train_one_epoch
from meta_model import EncModel
import torch.optim as optim
import os
import argparse
from reg_data_loader import *
from utils import get_saved_file
torch.manual_seed(123)
np.random.seed(1234)


def get_arguments():
    parser = argparse.ArgumentParser(description='ST-MAML')
    parser.add_argument('--num_val', type=int, default=10)
    parser.add_argument('--num_sample_function', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=3)
    parser.add_argument('--meta_bs', type=int, default=36)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--total_batches', type=int, default=100)
    parser.add_argument('--inner_loop_grad_clip', type=float, default=10.)
    parser.add_argument('--noise_std', default=0.3, type=float)
    parser.add_argument('--resume_train', default=False, action='store_true')
    parser.add_argument('--inner_lr', default=1e-3, type=float)
    parser.add_argument('--inner_step', default=3, type=int)
    parser.add_argument('--img_size', default=28, type=int)
    parser.add_argument('--loss_type', default='BCEloss', choices=['BCEloss', 'MSEloss'])
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--kl_weight', default=1., type=float)
    parser.add_argument('--in_weight_rest', default=1e-2, type=float)
    parser.add_argument('--aug_enc', default=False, action='store_true')
    parser.add_argument('--model_type', default='deter', choices=['prob', 'deter'])
    parser.add_argument('--act_type', default='softplus', choices=['softplus', 'sigmoid'])
    return parser.parse_args()


args = get_arguments()

print(args.resume_train)


dataset_list = ['mnist', 'fmnist', 'kmnist']

dataset_list = [get_dloader('./data/', dataset, args.meta_bs, args.num_sample_function, args.img_size) for dataset in dataset_list]

trainset = MultimodalFewShotDataset(
            dataset_list,
            num_total_batches=args.total_batches,
            mix_meta_batch=True,
            mix_mini_batch=False,
            txt_file=None,
            train=True,
        )
evalset = MultimodalFewShotDataset(
            dataset_list,
            num_total_batches=100,
            mix_meta_batch=True,
            mix_mini_batch=False,
            txt_file=None,
            train=True,
        )


if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.set_device(args.gpu_id)
else:
    device = 'cpu'
print(device)

state_str = os.path.join(args.output_folder, args.loss_type+'_'+str(args.inner_step)\
    +'_'+args.model_type+'_'+str(args.kl_weight)+'_'+str(args.in_weight_rest)+'_'+str(args.act_type)+'_'+str(args.aug_enc)\
        +'_'+str(args.inner_lr)+'_'+str(args.num_sample_function))

args.save_epoch_loss = os.path.join(state_str, 'save_epoch_loss.txt')
if not os.path.exists(args.output_folder):
    os.mkdir(args.output_folder)
if not os.path.exists(state_str):
    os.mkdir(state_str)

    
if not os.path.exists(args.save_epoch_loss):
    f = open(args.save_epoch_loss, 'a+')
    

  

model = EncModel(2, [64, 128, 256, 128], 1, 80, 128, 3, [20, 40],64, args.aug_enc, args.model_type, args.act_type)
model.to(device)


print(model)

if args.resume_train:
    path = state_str
    start_epoch = get_saved_file(path)
    start_model_str = os.path.join(state_str, 'epoch_'+str(start_epoch)+'.pt')
    model.load_state_dict(torch.load(start_model_str, map_location='cuda:'+str(args.gpu_id)))
    start_epoch += 1


else:
    start_epoch=0
    



opt = optim.Adam(model.parameters(), lr=args.lr)



save_loss = 100000
for epoch in range(start_epoch, args.epoch):
    meta_train_one_epoch(model, trainset, opt, epoch, args, 'train', device)
    _, eval_loss = meta_eval_one_epoch(model, evalset, epoch, args)
    if eval_loss < save_loss:
        save_loss = eval_loss
        torch.save(model.state_dict(), os.path.join(state_str, 'best.pt'))
    if (epoch)%1==0:
        torch.save(model.state_dict(), os.path.join(state_str, 'epoch_'+str(epoch)+'.pt'))
        
        
