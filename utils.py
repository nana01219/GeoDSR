import os
import glob
import tqdm
import random
import tensorboardX

import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from apex import amp

from PIL import Image
import csv

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g. 
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def visualize_2d(x, batched=False, renormalize=False):
    # x: [B, 3, H, W] or [B, 1, H, W] or [B, H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if batched:
        x = x[0]
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    
    if len(x.shape) == 3:
        if x.shape[0] == 3:
            x = x.transpose(1, 2, 0) # to channel last
        elif x.shape[0] == 1:
            x = x[0] # to grey
        
    print(f'[VISUALIZER] {x.shape}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    if len(x.shape) == 3:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.matshow(x)
    plt.show()


class RMSEMeter:
    def __init__(self, args):
        self.args = args
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, data, preds, truths, csv_writer=None, eval=False):
        preds, truths = self.prepare_inputs(preds, truths) # [B, 1, H, W]

        if eval:
            B, C, H, W = data['hr_image'].shape
            preds = preds.reshape(B, 1, H, W)
            truths = truths.reshape(B, 1, H, W)

            # clip borders (reference: https://github.com/cvlab-yonsei/dkn/issues/1)
            preds = preds[:, :, 6:-6, 6:-6]
            truths = truths[:, :, 6:-6, 6:-6]
            
        # rmse
        rmse = np.sqrt(np.mean(np.power(preds - truths, 2)))
        
        # to report per-image rmse 
        if self.args.report_per_image:
            print('rmse = ', rmse)
            if not csv_writer:
                print("ERROR:no writer")
            csv_writer.writerow([str(self.N), str(rmse)])

        self.V += rmse
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "rmse"), self.measure(), global_step)

    def report(self):
        return f'RMSE = {self.measure():.6f}'


class Trainer(object):
    def __init__(self, 
                 args,
                 name, # name of this experiment
                 model, # network 
                 objective=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 opt_level='O0', # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=1, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=False, # use loss as the first metirc
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 pre_trained_model=None, #这个是读取指定的pretrained model
                 ):
        
        self.args = args
        self.name = name
        self.mute = mute
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.opt_level = opt_level
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.pre_trained_model = pre_trained_model
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        if isinstance(self.objective, nn.Module):
            self.objective.to(self.device)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opt_level, verbosity=0)

        self.sum_t = 0

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}_best.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Model randomly initialized ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    print(self.best_path)
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    print(self.best_path)
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    os._exit()
                    self.load_checkpoint()
            elif self.use_checkpoint == "from":
                the_path = f"{self.ckpt_path}/{self.pre_trained_model}"
                print(the_path)
                if os.path.exists(the_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(the_path)
                else:
                    self.log(f"[INFO] {the_path} not found...")
                    os._exit()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    def log(self, *args):
        if self.local_rank == 0:
            if not self.mute: 
                print(*args)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)    

    ### ------------------------------	

    def train_step(self, data):
        # print('--------------------------------------------------------')
        gt = data['hr_depth']
        # a = time.time()
        pred = self.model(data)
        # b = time.time()
        # self.sum_t += (b-a)
        # print("loss--", gt.shape, pred.shape)
        loss = self.objective(pred, gt)

        # rescale
        # max和min是用来归一化的
        # print(pred.shape)    # 1, 30720, 1
        # print(gt.shape)      # 1, 30720, 1
        # print(data['lr'].shape) # 1, 1, 32, 32
        pred = pred * (data['max'] - data['min']) + data['min']
        gt = gt * (data['max'] - data['min']) + data['min']
        
        return pred, gt, loss

    def eval_step(self, data):
        return self.train_step(data)

    def test_step(self, data):  
        # print('HR', data['hr_image'].shape)
        # print(data.keys())
        B, C, H, W = data['hr_image'].shape
        gt = data['hr_depth']
        input = data["lr_depth"]
        a = time.time()
        pred = self.model(data)
        b = time.time()
        dtime = b-a
        pred = pred * (data['max'] - data['min']) + data['min']
        gt = gt * (data['max'] - data['min']) + data['min']
        input = input * (data['max'] - data['min']) + data['min']
        # print(type(gt), type(pred))
        print(pred.shape, gt.shape)
        # pred = pred.reshape(B, 1, H, W)
        # gt = gt.reshape(B, 1, H, W)


        #visualize_2d(data['image'], batched=True)
        #visualize_2d(data['lr'], batched=True)
        #visualize_2d(pred, batched=True)

        return pred, gt, dtime, input

    ### ------------------------------

    def train(self, train_loader_fix, train_loader_vary, valid_loader, max_epochs, AR_epoch = None):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        
        if AR_epoch < max_epochs and AR_epoch > 0:
            self.two_stage = True

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            # 训练
            if AR_epoch >= epoch:
                self.train_one_epoch(train_loader_fix)
                # 保存
                if self.workspace is not None and self.local_rank == 0:
                    self.save_checkpoint_fix(full=True, best=False)
                # 验证
                if self.epoch % self.eval_interval == 0 or self.epoch + 15 > max_epochs:
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint_fix(full=False, best=True)
            else:
                self.train_one_epoch(train_loader_vary)
                 # 保存
                if self.workspace is not None and self.local_rank == 0:
                    self.save_checkpoint(full=True, best=False)

                # 验证
                if self.epoch % self.eval_interval == 0 or self.epoch + 15 > max_epochs:
                    self.evaluate_one_epoch(valid_loader)
                    self.save_checkpoint(full=False, best=True)


            if AR_epoch == epoch:
                self.stats["best_result"] = None
                self.optimizer.param_groups[0]['lr'] = 0.0001
                self.lr_scheduler.last_epoch = 0

           
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader):
        #if os.path.exists(self.best_path):
        #    self.load_checkpoint(self.best_path)
        #else:
        #    self.load_checkpoint()
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader)
        self.use_tensorboardX = use_tensorboardX


    def test(self, loader, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results', f'{self.name}_{self.args.dataset}_{self.args.scale}')
            gt_path = os.path.join(self.workspace, 'results', f'gt_{self.name}_{self.args.dataset}_{self.args.scale}')
            error_path = os.path.join(self.workspace, 'results', f'error_{self.name}_{self.args.dataset}_{self.args.scale}')
            input_path = os.path.join(self.workspace, 'results', f'input_{self.name}_{self.args.dataset}_{self.args.scale}')
            csv_path = os.path.join(self.workspace, 'results', f'{self.name}_{self.args.dataset}_{self.args.scale}.csv')
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(error_path, exist_ok=True)
        os.makedirs(input_path, exist_ok=True)
        f = open(csv_path, 'w', encoding='utf-8')
        csv_writer  =csv.writer(f)
        
        
        print(" this is a test ")
        self.log(f"==> Start Test, save results to {save_path}")


        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():

            all_time = 0
            for data in loader:
                
                data = self.prepare_data(data)
                preds, gt, dtime, input = self.test_step(data)
                all_time += dtime
                for metric in self.metrics:
                    metric.update(data, preds, gt, csv_writer)

                preds = preds.detach().cpu().numpy() # [B, 1, H, W]
                gt = gt.detach().cpu().numpy() # [B, 1, H, W]
                input = input.detach().cpu().numpy()

                for b in range(preds.shape[0]):
                    idx = data['idx'][b]
                    if not isinstance(idx, str):
                        idx = str(idx.item())
                    pred = preds[b][0]
                    gt = gt[b][0]
                    input = input[b][0]
                    error = abs(gt-pred)
                    # error = np.sqrt(error)
                    aa = np.min(gt)
                    bb = np.max(gt)

                    # print(pred.shape, gt.shape)
                    plt.imsave(os.path.join(save_path, self.name+idx+'.png'), pred,vmin=aa,vmax=bb,  cmap="jet")
                    plt.imsave(os.path.join(gt_path, "gt"+idx+'.png'), gt, vmin=aa,vmax=bb, cmap='jet')
                    plt.imsave(os.path.join(input_path, "input"+idx+'.png'), input, vmin=aa,vmax=bb, cmap='jet')
                    plt.imsave(os.path.join(error_path,  self.name+'error'+idx+'.png'), error,vmin=5,vmax=50,  cmap="plasma")
                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.") 
        self.log(metric.report())
        print("total time, pics, average time:", all_time, len(loader), all_time/len(loader))

        # print(error.mean())  
        # print(error.max())

    # 转移到cuda
    def prepare_data(self, data):
        # print(type(data))
        if isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, np.ndarray):
                    data[i] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[i] = v.to(self.device)
        # 基本可以判断，dataloader输出的是dict，而且是大的dict  
        elif isinstance(data, dict):
            for k, v in data.items():
                # print(k)
                if isinstance(v, np.ndarray):
                    data[k] = torch.from_numpy(v).to(self.device)
                if torch.is_tensor(v):
                    data[k] = v.to(self.device)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        else: # is_tensor
            data = data.to(self.device)

        return data

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']} ... <====================")

        total_loss = []
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        # criterion = nn.MSELoss()
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            data = self.prepare_data(data)
            # 前向并计算loss
            preds, truths, loss = self.train_step(data)
            # loss *= 100


            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss.append(loss.item())
            if self.local_rank == 0:
                for metric in self.metrics:
                    metric.update(data, preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={total_loss[-1]:.4f}, lr={self.optimizer.param_groups[0]['lr']}")
                else:
                    pbar.set_description(f'loss={total_loss[-1]:.4f}')
                pbar.update(loader.batch_size * self.world_size)

        average_loss = np.mean(total_loss)
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            for metric in self.metrics:
                self.log(metric.report())
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"average_loss={average_loss:.4f}")
        torch.cuda.empty_cache()


    def evaluate_one_epoch(self, loader):
        self.log(f"<------- {self.epoch} -------> Evaluate at epoch  ")
        self.sum_t = 0

        total_loss = []
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                data = self.prepare_data(data)
                # a = time.time()
                preds, truths, loss = self.eval_step(data)
                # b = time.time()
                # self.sum_t += b-a
                
                total_loss.append(loss.item())            
                if self.local_rank == 0:
                    for metric in self.metrics:
                        metric.update(data, preds, truths, eval=True)

                    pbar.set_description(f'loss={total_loss[-1]:.4f}')
                    pbar.update(loader.batch_size * self.world_size)

        average_loss = np.mean(total_loss)
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report())   # 报RMSE的信息
                if self.use_tensorboardX:   
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        self.log(f"val loss = {average_loss:.4f}")
        print("-------> self.sum_t=", self.sum_t, self.sum_t/len(loader))
        self.sum_t=0

    def save_checkpoint(self, full=True, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
            'model': self.model.state_dict(),
        }

        if full:
            state['amp'] = amp.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        # 如果不是要保存best
        if not best:

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            # 新的epoch入栈
            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                # 旧的epoch出栈，移除旧的，保存新的
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]
                    torch.save(state, self.best_path)
            else:
                self.log(f"[INFO] no evaluated results found, skip saving best checkpoint.")

    def save_checkpoint_fix(self, full=True, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
            'model': self.model.state_dict(),
        }

        if full:
            state['amp'] = amp.state_dict()
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        # 如果不是要保存best
        if not best:

            file_path = f"{self.ckpt_path}/{self.name}_fix_ep{self.epoch:04d}.pth.tar"

            # 新的epoch入栈
            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                # 旧的epoch出栈，移除旧的，保存新的
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            torch.save(state, file_path)

        else:    
            file_path = f"{self.ckpt_path}/{self.name}_fix_best.pth.tar"
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]
                    torch.save(state, file_path)
            else:
                self.log(f"[INFO] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
            else:
                self.log("[INFO] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            return

        self.model.load_state_dict(checkpoint_dict['model'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        if self.use_checkpoint == 'from':
            # self.epoch = 1
            # self.optimizer.param_groups[0]['lr'] = 0.00008579218380547904
            self.stats["best_result"] = checkpoint_dict['best_result']
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer. Skipped.")
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler. Skipped.")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer. Skipped.")
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler. Skipped.")
                
        if 'amp' in checkpoint_dict:
            amp.load_state_dict(checkpoint_dict['amp'])
            self.log("[INFO] loaded amp.")