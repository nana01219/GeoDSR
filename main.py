import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main.py  --test --checkpoint best  --name Stage=lr-1=2layDSF=scale=RCAN_modulation_v2=16_fixed_fix --model JIIF  --dataset NYU  --scale 16

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F

from utils import *
from datasets import *
from models import *

def get_dataset(name, data):
    if 'geo' in name or 'Geo' in name or "GASA" in name:
        if data == 'NYU':
            dataset = Geo_NYUDataset
        elif data == 'Middlebury':
            dataset = Geo_MiddleburyDataset
        elif data == 'Lu':
            dataset = Geo_LuDataset
        elif data == 'Luz':
            dataset = Geo_LuzDataset
        else:
            raise NotImplementedError(f'Dataset {name} not found')
    else:
        if data == 'NYU':
            dataset = NYUDataset
        elif data == 'Lu':
            dataset = LuDataset
        elif data == 'Middlebury':
            dataset = MiddleburyDataset
        elif data == 'Luz':
            dataset = LuzDataset
        else:
            raise NotImplementedError(f'Dataset {name} not found')
    return dataset

parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, default='Mod_v2_2023=DSF_real60-0.2_b8_p256_c128_2GeoOP_vRestormer_ESA_res+res=8')
parser.add_argument('--name', type=str, default='GASA_e400_s8')
parser.add_argument('--epoch', default=400, type=int, help='max epoch')
parser.add_argument('--AR_epoch',  default=200, type=int, help='epochs for the first stage')
parser.add_argument('--eval_interval',  default=10, type=int, help='eval interval')
parser.add_argument('--checkpoint',  default='scratch', type=str, help='checkpoint to use')
parser.add_argument('--scale',  default=8, type=int, help='scale')
parser.add_argument('--interpolation',  default='bicubic', type=str, help='interpolation method to generate lr depth')
parser.add_argument('--lr',  default=0.0001, type=float, help='learning rate')
# parser.add_argument('--lr_step',  default=1, type=float, help='learning rate decay step')
# parser.add_argument('--lr_gamma',  default=0.975, type=float, help='learning rate decay gamma')
parser.add_argument('--lr_step',  default=60, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=0.2, type=float, help='learning rate decay gamma')
parser.add_argument('--input_size',  default=256, type=int, help='crop size for hr image')
parser.add_argument('--block_num',  default=4, type=int, help='')
parser.add_argument('--rgb_c',  default=1, type=int, help='')
parser.add_argument('--depth_c',  default=2, type=int, help='')
parser.add_argument('--test_val',  default=1, type=int, help='')
parser.add_argument('--sample_q',  default=30720, type=int, help='')
parser.add_argument('--model', type=str, default='GASA')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--seed', type=int, default=2341)
parser.add_argument('--dataset', type=str, default='NYU')
parser.add_argument('--data_root', type=str, default="./data/nyu_labeled/")
parser.add_argument('--train_batch', type=int, default=1)
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--pre_trained_model',  default='Stage_lr-1=2layDSF=biasxSE_RBF_256=8_fix_best.pth.tar', type=str, help='the specific model to load')
parser.add_argument('--test',  action='store_true', help='test mode')
parser.add_argument('--report_per_image',  action='store_true', help='report RMSE of each image')
parser.add_argument('--save',  action='store_true', help='save results')
parser.add_argument('--batched_eval',  action='store_true', help='batched evaluation to avoid OOM for large image resolution')


args = parser.parse_args()

seed_everything(args.seed)
print("****************************")
print("*****AR epoch: ", args.AR_epoch, "/",  args.epoch, "*****")
print("****************************")

# model
if args.model == 'GASA':
    model = GASA_model(args,  feat_dim = args.depth_c, guide_dim = args.rgb_c, block_num =args.block_num )
else:
    raise NotImplementedError(f'Model {args.model} not found')

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
dataset = get_dataset(args.name, args.dataset)

if args.model in ['GASA']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation, augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, if_AR=False)
        train_dataset_vary = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation, augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size,  if_AR=True)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None) # full image
else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    train_loader_1 = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers)
    train_loader_2 = torch.utils.data.DataLoader(train_dataset_vary, batch_size=args.train_batch, pin_memory=False, drop_last=False, shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, pin_memory=False, drop_last=False, shuffle=False, num_workers=args.num_workers)

# trainer
if not args.test:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval, pre_trained_model=args.pre_trained_model)
else:
    trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, pre_trained_model=args.pre_trained_model)

# main
if not args.test:
    trainer.train(train_loader_1, train_loader_2,  test_loader, args.epoch, args.AR_epoch)

if args.save:
    # save results (doesn't need GT)
    trainer.test(test_loader)
else:
    # evaluate (needs GT)
    trainer.evaluate(test_loader)

if not args.save:
    val_trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint="best", pre_trained_model=args.pre_trained_model)
    if not args.test and (args.test_val > 0):
        scale_list = [4, 8, 16]
        data_list = ["NYU","Middlebury","Lu","Luz"]
        root_list = ["./data/nyu_labeled/","./data/depth_enhance/01_Middlebury_Dataset","./data/depth_enhance/03_RGBD_Dataset","./data/depth_enhance/02_RGBZ_Dataset"]
        for num, j in enumerate(data_list):
            dataset = get_dataset(args.name, j)
            rr = root_list[num]
            for i in scale_list:
                val_set = dataset(root=rr, split='test', scale=i, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch, pin_memory=False, drop_last=False, shuffle=False, num_workers=args.num_workers)
                print("--------------- scale:", i, "--dataset:", j, "----------------")
                trainer.evaluate(val_loader)
                




        
            

            

