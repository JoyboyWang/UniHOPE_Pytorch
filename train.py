import argparse
import os
import torch
import numpy as np

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from functools import partial
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored

from tqdm import tqdm
from data_loader.data_loader import fetch_dataloader
from model.fetch_model import fetch_model
from optimizer.optimizer import fetch_optimizer, fetch_optimizer_with_params_same_lr
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='', type=str, help='Directory containing params.json')
parser.add_argument('--resume', default=None, type=str, help='Path of model weights')
parser.add_argument('--debug', '-d', action='store_true', help='Debug')
parser.add_argument('--only_weights', '-ow', action='store_true', help='Only load model weights or load all train status')
parser.add_argument("-v", "--verbose", action="store_true", help="Output the loss at each iteration")
parser.add_argument('--skip_evaluate', action='store_true', help='Whether skip evaluation process to avoid NCCL erros')

def get_learning_rate(epoch, step, base_lr, minibatch_per_epoch, warmup_epoch, stop_epoch):
    final_lr = 0.0
    warmup_iter = minibatch_per_epoch * warmup_epoch
    warmup_lr_schedule = np.linspace(0, base_lr, warmup_iter)
    decay_iter = minibatch_per_epoch * (stop_epoch - warmup_epoch)
    if epoch < warmup_epoch:
        cur_lr = warmup_lr_schedule[step + epoch*minibatch_per_epoch]
    else:
        if epoch < stop_epoch // 2:
            return base_lr

        return base_lr / 10
        
    return cur_lr

class Trainer():

    def __init__(self, cfg):
        # Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        self.accelerator = Accelerator(split_batches=True, mixed_precision='no', kwargs_handlers=[ddp_kwargs])

        # Config status
        self.cfg = cfg

        # Set logger
        self.logger = tool.set_logger(os.path.join(cfg.base.model_dir, 'train.log'))

        # Fetch dataloader
        self.logger.info(f'Dataset: {cfg.data.name}')
        self.dl, self.ds = fetch_dataloader(cfg)
        
        # added by yqwang
        desc = self.cfg.base.get('desc', None)
        if desc is not None:
            self.logger.info(f'Exp Desc: {desc}')
        
        self.logger.info(self.cfg)

        # Fetch model
        self.model = fetch_model(cfg.model.name, cfg)

        # Define optimizer and scheduler
        if cfg.model.name == 'kypt_transformer':
            # follow the original code
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if ("backbone_net" not in n and 'decoder_net' not in n) and p.requires_grad]},
                {
                    "params": [p for n, p in self.model.named_parameters() if ("backbone_net" in n or 'decoder_net' in n) and p.requires_grad],
                    "lr": cfg.optimizer.lr*0.1,
                },
            ]
            self.optimizer, self.scheduler = fetch_optimizer_with_params_same_lr(cfg, self.model, param_dicts)
        else:
            self.optimizer, self.scheduler = fetch_optimizer(cfg, self.model)

        # Prepare model, dataloader, optimizer, and scheduler with accelerator
        for k, v in self.dl.items():
            self.dl[k] = self.accelerator.prepare(v)
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        # Init some recorders
        self.init_status()
        self.init_tb()

    def init_status(self):
        self.epoch = 0
        self.step = 0
        # Train status: model, optimizer, scheduler, epoch, step
        self.train_status = {}
        # Loss status
        self.loss_status = defaultdict(tool.AverageMeter)
        # Metric status: val, test
        self.metric_status = defaultdict(lambda: defaultdict(tool.AverageMeter))
        # Score status: val, test
        self.score_status = {}
        for split in ['val', 'test']:
            self.score_status[split] = {'cur': np.inf, 'best': np.inf}

    def init_tb(self):
        # Tensorboard
        loss_tb_dir = os.path.join(self.cfg.base.model_dir, 'summary/loss')
        os.makedirs(loss_tb_dir, exist_ok=True)
        self.loss_writter = SummaryWriter(log_dir=loss_tb_dir)
        metric_tb_dir = os.path.join(self.cfg.base.model_dir, 'summary/metric')
        os.makedirs(metric_tb_dir, exist_ok=True)
        self.metric_writter = SummaryWriter(log_dir=metric_tb_dir)

    def update_step(self):
        if not self.accelerator.is_main_process:
            return
        self.step += 1

    def update_epoch(self):
        if not self.accelerator.is_main_process:
            return
        self.epoch += 1

    def update_loss_status(self, loss, batch_size):
        for k, v in loss.items():
            self.loss_status[k].update(val=v.item(), num=batch_size)

    def update_metric_status(self, metric, split, batch_size):
        if not self.accelerator.is_main_process:
            return
        for k, v in metric.items():
            self.metric_status[split][k].update(val=v.item(), num=batch_size)
            self.score_status[split]['cur'] = self.metric_status[split][self.cfg.metric.major_metric].avg

    def reset_loss_status(self):
        if not self.accelerator.is_main_process:
            return
        for k, v in self.loss_status.items():
            self.loss_status[k].reset()

    def reset_metric_status(self, split):
        if not self.accelerator.is_main_process:
            return
        for k, v in self.metric_status[split].items():
            self.metric_status[split][k].reset()

    def tqdm_info(self, split):
        if split == 'train':
            exp_name = self.cfg.base.model_dir.split('/')[-1]
            print_str = f'{exp_name}, E:{self.epoch:3d}, lr:{self.scheduler.get_last_lr()[0]:.2E}, '
            print_str += f'loss: {self.loss_status["total"].val:.4g}/{self.loss_status["total"].avg:.4g}'
        else:
            print_str = ''
            for k, v in self.metric_status[split].items():
                print_str += f'{k}: {v.val:.4g}/{v.avg:.4g}'
        return print_str

    def print_metric(self, split, only_best=False):
        if not self.accelerator.is_main_process:
            return
        is_best = self.score_status[split]['cur'] < self.score_status[split]['best']
        color = 'white' if split == 'val' else 'red'
        print_str = ' | '.join(f'{k}: {v.avg:.4g}' for k, v in self.metric_status[split].items())
        if only_best:
            if is_best:
                self.logger.info(colored(f'Best Epoch: {self.epoch}, {split} Results: {print_str}', color, attrs=['bold']))
        else:
            self.logger.info(colored(f'Epoch: {self.epoch}, {split} Results: {print_str}', color, attrs=['bold']))

    def write_loss_to_tb(self, split):
        if not self.accelerator.is_main_process:
            return
        if self.step % self.cfg.summary.save_summary_steps == 0:
            for k, v in self.loss_status.items():
                self.loss_writter.add_scalar(f'{split}_loss/{k}', v.val, self.step)

    def write_metric_to_tb(self, split):
        if not self.accelerator.is_main_process:
            return
        for k, v in self.metric_status[split].items():
            self.metric_writter.add_scalar(f'{split}_metric/{k}', v.avg, self.epoch)

    def write_custom_info_to_tb(self, input, output, split):
        if not self.accelerator.is_main_process:
            return
        pass

    def save_ckpt(self):
        if not self.accelerator.is_main_process:
            return
        # Save latest and best metrics
        for split in ['val', 'test']:
            if split not in self.dl:
                continue
            latest_metric_path = os.path.join(self.cfg.base.model_dir, f'{split}_metric_latest.json')
            tool.save_dict_to_json(self.metric_status[split], latest_metric_path)
            is_best = self.score_status[split]['cur'] < self.score_status[split]['best']
            if is_best:
                self.score_status[split]['best'] = self.score_status[split]['cur']
                best_metric_path = os.path.join(self.cfg.base.model_dir, f'{split}_metric_best.json')
                tool.save_dict_to_json(self.metric_status[split], best_metric_path)

        # Model states
        state = {
            'state_dict': self.accelerator.get_state_dict(self.model),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'score_status': self.score_status
        }

        # Save middle checkpoint
        if self.epoch % self.cfg.summary.save_mid_freq == 0:
            middle_ckpt_path = os.path.join(self.cfg.base.model_dir, f'model_{self.epoch}.pth')
            torch.save(state, middle_ckpt_path)

        # Save latest checkpoint
        if self.epoch % self.cfg.summary.save_latest_freq == 0:
            latest_ckpt_path = os.path.join(self.cfg.base.model_dir, 'model_latest.pth')
            torch.save(state, latest_ckpt_path)

        # Save latest and best checkpoints
        for split in ['val', 'test']:
            if split not in self.dl:
                continue
            # Above code has updated the best score to cur
            is_best = self.score_status[split]['cur'] == self.score_status[split]['best']
            if is_best:
                self.logger.info(f'Current is {split} best, score={self.score_status[split]["best"]:.3f}')
                # Save best checkpoint
                if self.epoch > self.cfg.summary.save_best_after:
                    best_ckpt_path = os.path.join(self.cfg.base.model_dir, f'{split}_model_best.pth')
                    torch.save(state, best_ckpt_path)

    def load_ckpt(self):
        accelerator = self.accelerator
        device = accelerator.device

        state = torch.load(self.cfg.base.resume, map_location=device)

        ckpt_component = []

        if 'state_dict' in state and self.model is not None:
            model = self.accelerator.unwrap_model(self.model)
            missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
            self.logger.info("missing_keys: {}".format(len(missing_keys)))
            self.logger.info("unexpected_keys: {}".format(len(unexpected_keys)))
            ckpt_component.append('net')
            
            # for H2ONet, do not load the weight for H2ONet_Decoder.rot_reg
            if self.cfg.model.name == "h2onet" and self.cfg.train.get("not_load_rot_reg", False):
                # import ipdb; ipdb.set_trace()
                model.decoder3d.init_weights()  # initialize the rot_reg module
            
            old_parameters = []
            new_parameters = []
            
            old_param_idx = []
            new_param_idx = []
            
            # new_modules = []
            for i, (name, param) in enumerate(model.named_parameters()):
                if name not in missing_keys:
                    old_parameters.append(param)
                    old_param_idx.append(i)
                else:
                    new_parameters.append(param)
                    new_param_idx.append(i)
                        
            # freeze the old parameters if required
            freeze_model = self.cfg.train.get("freeze_model", False)
            if freeze_model and len(missing_keys) > 0:  # only freeze the old parameters when there're new modules
                # print(missing_keys)  # ['feature_mapper.0.w_qs.weight', 'feature_mapper.0.w_ks.weight', 'feature_mapper.0.w_vs.weight', 'feature_mapper.0.fc.weight', 'feature_mapper.0.layer_norm.weight', 'feature_mapper.0.la$er_norm.bias', 'feature_mapper.1.weight', 'feature_mapper.1.bias']
                for param in old_parameters:
                    param.requires_grad = False

        if not self.cfg.base.only_weights:
            if 'optimizer' in state and self.optimizer is not None:
                if len(missing_keys) == 0:
                    self.optimizer.load_state_dict(state['optimizer'])  # do not allow new parameters in network
                else:
                    saved_optimizer_state_dict = state['optimizer']
                    
                    # Current optimizer state dict and its structure
                    current_optimizer_state_dict = self.optimizer.state_dict()

                    # Filter the saved optimizer state dict to include only the parameters that exist in the current model
                    filtered_state_dict = {
                        "state": {},
                        # "state": saved_optimizer_state_dict["state"],
                        "param_groups": current_optimizer_state_dict["param_groups"],  # Copy current param_groups
                    }

                    # Populate the filtered state with states only for the matching parameters
                    # for saved_param, saved_value in saved_optimizer_state_dict["state"].items():
                    # Find the parameter in the current model that matches this saved state
                    saved_state_dict = saved_optimizer_state_dict["state"]
                    for i, param in enumerate(old_param_idx):
                        filtered_state_dict["state"][param] = saved_state_dict[i]
                    
                    for i, param in enumerate(new_param_idx):
                        filtered_state_dict["state"][param] = {"step": 0, "exp_avg": torch.zeros_like(new_parameters[i].data), "exp_avg_sq": torch.zeros_like(new_parameters[i].data)}
                    
                    # Load the filtered state dict into the optimizer
                    self.optimizer.load_state_dict(filtered_state_dict)
                    
                ckpt_component.append('opt')
                
            if 'scheduler' in state and self.scheduler is not None:
                self.scheduler.load_state_dict(state['scheduler'])
                ckpt_component.append('sch')

            if 'step' in state:
                self.step = state['step']
                ckpt_component.append('step')

            if 'epoch' in state:
                self.epoch = state['epoch']
                ckpt_component.append('epoch')

            if 'score_status' in state:
                self.score_status = state['score_status']
                ckpt_component.append(f'score status: {self.score_status}')

        ckpt_component = ', '.join(i for i in ckpt_component)
        self.logger.info(f'Loaded models from: {self.cfg.base.resume}')
        self.logger.info(f'Ckpt load: {ckpt_component}')

    def train_and_evaluate(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.logger.info(f'Starting training for {self.cfg.train.num_epochs} epoch(s)')
        # Load weights from restore_file if specified
        if self.cfg.base.resume is not None:
            self.load_ckpt()
        # test_weight = None
        for epoch in range(self.epoch, self.cfg.train.num_epochs):
            # Train one epoch
            # Reset loss status
            self.reset_loss_status()
            # Set model to training mode
            torch.cuda.empty_cache()
            self.model.train()
            # Use tqdm for progress bar
            t = tqdm(dynamic_ncols=True, total=len(self.dl['train']), disable=not accelerator.is_main_process)
            # Train loop
            iter = 0
            for batch_idx, batch_input in enumerate(self.dl['train']):
                
                if self.cfg.model.name == 'simple_hand':
                    # adopt the warmup strategy provided in SimpleHand
                    lr = get_learning_rate(epoch, iter, self.cfg.optimizer.base_lr, self.cfg.data.minibatch_per_epoch, self.cfg.optimizer.warmup_epoch, self.cfg.train.num_epochs)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr

                if self.cfg.base.debug and batch_idx >= 10:
                    break

                # Move input to GPU if available
                batch_input = tool.tensor_gpu(batch_input, device=device)
                
                # Compute model output and loss
                if self.cfg.model.name == 'kypt_transformer':
                    batch_output = self.model(batch_input, epoch)
                else:
                    batch_output = self.model(batch_input)
                
                loss = compute_loss(self.cfg, batch_input, batch_output, epoch)
                
                self.accelerator.backward(loss['total'])
                # Update loss status and print current loss and average loss
                # Get real batch size
                if 'img' in batch_input:
                    batch_size = batch_input['img'].size()[0]
                else:
                    batch_size = self.cfg.train.batch_size
                self.update_loss_status(loss=loss, batch_size=batch_size)

                # add gradient clip
                if 'grad_norm_clip' in self.cfg.train and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_norm_clip)

                accelerator.wait_for_everyone()

                # Clean previous gradients, compute gradients of all variables wrt loss
                self.optimizer.step()
                
                self.optimizer.zero_grad()

                accelerator.wait_for_everyone()

                # Update step: step += 1
                self.update_step()
                # Write loss to tensorboard
                self.write_loss_to_tb(split='train')
                # Write custom info to tensorboard
                self.write_custom_info_to_tb(batch_input, batch_output, split='train')
                # Training info print
                print_str = self.tqdm_info(split='train')
                # Tqdm settings
                t.set_description(desc=print_str)
                t.update()
                
                iter += 1
                
            # Close tqdm
            t.close()

            if not self.cfg.base.skip_evaluate:
                # Evaluate one epoch, check if current is best, save best and latest checkpoints
                with torch.no_grad():
                    # Set model to evaluation mode
                    torch.cuda.empty_cache()
                    self.model.eval()
                    # Compute metrics over the dataset
                    for split in ['val', 'test']:
                        if split not in self.dl:
                            continue
                        # Use tqdm for progress bar
                        t = tqdm(dynamic_ncols=True, total=len(self.dl[split]), disable=not accelerator.is_main_process)
                        # Initialize loss and metric statuses
                        self.reset_loss_status()
                        self.reset_metric_status(split)
                        cur_sample_idx = 0
                        for batch_idx, batch_input in enumerate(self.dl[split]):
                            
                            if self.cfg.base.debug and batch_idx >= 10:
                                break

                            # Move data to GPU if available
                            batch_input = tool.tensor_gpu(batch_input, device=device)
                            # Compute model output
                            batch_output = self.model(batch_input)
                            
                            # Compute all metrics on this batch
                            for k, v in batch_output.items():
                                if type(v) is not np.ndarray and v is not None and not v.is_contiguous():  # I removed numpy
                                    batch_output[k] = v.contiguous()  # tensor must be contiguous for gather operation
                            
                            batch_input, batch_output = accelerator.gather_for_metrics((batch_input, batch_output))  # NOTE: if return > batch size will throw error
                            
                            # Get real batch size
                            batch_size = batch_input['img'].size()[0]
                            metric = compute_metric(self.cfg, batch_input, batch_output)
                            self.update_metric_status(metric, split, batch_size)
                            # Tqdm settings
                            t.set_description(desc=f'{split} running')
                            t.update()
                            
                        # Close tqdm
                        t.close()

                        # Update data to tensorboard
                        self.write_metric_to_tb(split)
                        # # Write custom info to tensorboard
                        # mng.write_custom_info_to_tb(batch_input, batch_output, split)
                        # For each epoch, update and print the metric
                        self.print_metric(split, only_best=False)

            if self.cfg.model.name != "simple_hand":
                # Update scheduler
                self.scheduler.step()
            # Update epoch: epoch += 1
            self.update_epoch()
            # Save checkpoint
            self.save_ckpt()


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'cfg.json')
    assert os.path.isfile(json_path), f'No json configuration file found at {json_path}'
    cfg = Config(json_path).cfg

    # Update args into cfg.base
    cfg.base.update(vars(args))
    
    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    trainer = Trainer(cfg=cfg)
    trainer.train_and_evaluate()
    
    print("#Finish#")  # end flag for the inspector
