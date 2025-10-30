import argparse
import os
import torch
import json
from tqdm import tqdm
from data_loader.data_loader import fetch_test_dataloader
from model.fetch_model import fetch_model
from loss.loss import compute_loss, compute_metric
from common import tool
from common.manager import Manager
from common.config import Config
import numpy as np
from collections import OrderedDict
import eval
from torch.nn.parallel import DataParallel as DP
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="", type=str, help="Directory containing params.json")
parser.add_argument("--resume", default="", type=str, help="Path of model weights")
parser.add_argument("--resume_cls", default=None, type=str, help="Path of classification model weights")
parser.add_argument("--resume_h", default=None, type=str, help="Path of hand model weights")
parser.add_argument("--resume_ho", default=None, type=str, help="Path of hand-object model weights")
parser.add_argument("--debug", "-d", action="store_true", help="Debug")

def compute_hand_metric(model, mng: Manager):
    # Set model to evaluation mode
    torch.cuda.empty_cache()
    model.eval()

    h_list = [[],[],[],[]]  # joints_metric, joints_metric_aligned, verts_metric, verts_metric_aligned
    ho_list = [[],[],[],[]]  # joints_metric, joints_metric_aligned, verts_metric, verts_metric_aligned
    all_list = [[],[],[],[]]

    with torch.no_grad():
        # Compute metrics over the dataset
        for split in ["val", "test"]:
            if split not in mng.dataloader:
                continue
            # Initialize loss and metric statuses
            mng.reset_loss_status()
            mng.reset_metric_status(split)
            # Use tqdm for progress bar
            t = tqdm(total=len(mng.dataloader[split]))
            for batch_idx, batch_input in enumerate(mng.dataloader[split]):
                
                if mng.cfg.base.debug and batch_idx >= 10:
                    break
                
                # Move data to GPU if available
                batch_input = tool.tensor_gpu(batch_input)

                # Compute model output
                batch_output = model(batch_input)
                # Get real batch size
                if "img" in batch_input:
                    batch_size = batch_input["img"].size()[0]
                elif "img_0" in batch_input:
                    batch_size = batch_input["img_0"].size()[0]
                else:
                    batch_size = mng.cfg.test.batch_size
                    
                if "DEX" in mng.cfg.data.name:
                    batch_input = tool.tensor_gpu(batch_input, check_on=False)
                    batch_input = [{k: v[bid] for k, v in batch_input.items()} for bid in range(batch_size)]
                    
                    batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    
                    pred_lst = mng.dataset[split].get_predictions(batch_input, batch_output)
                    for b in range(batch_size):  # for different sample
                        grasping_flag = bool(batch_input[b]['gt_grasping'][0])
                        if grasping_flag:
                            ho_list[0].append(pred_lst[0][b])
                            ho_list[1].append(pred_lst[1][b])
                            ho_list[2].append(pred_lst[2][b])
                            ho_list[3].append(pred_lst[3][b])
                        else:
                            h_list[0].append(pred_lst[0][b])
                            h_list[1].append(pred_lst[1][b])
                            h_list[2].append(pred_lst[2][b])
                            h_list[3].append(pred_lst[3][b])
                            
                        all_list[0].append(pred_lst[0][b])
                        all_list[1].append(pred_lst[1][b])
                        all_list[2].append(pred_lst[2][b])
                        all_list[3].append(pred_lst[3][b])

                else:
                    batch_input = tool.tensor_gpu(batch_input, check_on=False)
                    batch_input = [{k: v[bid] for k, v in batch_input.items()} for bid in range(batch_size)]
                    
                    batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    # evaluate
                    metric = mng.dataset[split].evaluate(batch_input, batch_output)

                # Tqdm settings
                t.set_description(desc="")
                t.update()
           
            t.close()

    print(len(ho_list[0]), ",", len(h_list[0]), ",", len(all_list[0]))


    ho_gt_xyz_list = np.stack(ho_list[2], axis=0)
    ho_gt_verts_list = np.stack(ho_list[3], axis=0)
    ho_pred_xyz_list = np.stack(ho_list[0], axis=0)
    ho_pred_verts_list = np.stack(ho_list[1], axis=0)
    
    h_gt_xyz_list = np.stack(h_list[2], axis=0)
    h_gt_verts_list = np.stack(h_list[3], axis=0)
    h_pred_xyz_list = np.stack(h_list[0], axis=0)
    h_pred_verts_list = np.stack(h_list[1], axis=0)
    
    all_gt_xyz_list = np.stack(all_list[2], axis=0)
    all_gt_verts_list = np.stack(all_list[3], axis=0)
    all_pred_xyz_list = np.stack(all_list[0], axis=0)
    all_pred_verts_list = np.stack(all_list[1], axis=0)


    ho_output_dir = osp.join(cfg.base.model_dir, 'ho')
    if not osp.exists(ho_output_dir):
        os.makedirs(ho_output_dir)
    eval.main(ho_gt_xyz_list, ho_gt_verts_list, ho_pred_xyz_list, ho_pred_verts_list, ho_output_dir)
    
    
    h_output_dir = osp.join(cfg.base.model_dir, 'h')
    if not osp.exists(h_output_dir):
        os.makedirs(h_output_dir)
    eval.main(h_gt_xyz_list, h_gt_verts_list, h_pred_xyz_list, h_pred_verts_list, h_output_dir)
    
    
    all_output_dir = osp.join(cfg.base.model_dir, 'all')
    if not osp.exists(all_output_dir):
        os.makedirs(all_output_dir)
    eval.main(all_gt_xyz_list, all_gt_verts_list, all_pred_xyz_list, all_pred_verts_list, all_output_dir)

def main(cfg):
    # Set rank and is_master flag
    cfg.base.only_weights = False
    # Set the logger
    logger = tool.set_logger(os.path.join(cfg.base.model_dir, "test.log"))
    # Print GPU ids
    # gpu_ids = ", ".join(str(i) for i in [j for j in range(cfg.base.num_gpu)])
    # logger.info("Using GPU ids: [{}]".format(gpu_ids))
    # Fetch dataloader
    cfg.data.eval_type = ["test"]
    dl, ds = fetch_test_dataloader(cfg)
    # Fetch model
    model = fetch_model(cfg.model.name, cfg)
    
    if cfg.base.cuda:
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            torch.cuda.set_device(0)

        model = model.cuda()
        model = DP(model)
    
    # Initialize manager
    mng = Manager(model=model, optimizer=None, scheduler=None, cfg=cfg, dataloader=dl, dataset=ds, logger=logger)
    # Test the model
    mng.logger.info("Starting test.")
    
    # Load weights from restore_file if specified
    if mng.cfg.base.resume is not None:
        mng.load_ckpt()
        
    elif mng.cfg.base.resume_cls is not None and mng.cfg.base.resume_h is not None and mng.cfg.base.resume_ho is not None:
        eval.load_two_in_one_cls_weights(model, mng.logger, resume_h=mng.cfg.base.resume_h, resume_ho=mng.cfg.base.resume_ho, resume_cls=mng.cfg.base.resume_cls)
        
    else:
        raise NotImplementedError
        
    compute_hand_metric(model, mng)


if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "cfg.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    cfg = Config(json_path).cfg
    # Update args into cfg.base
    cfg.base.update(vars(args))
    
    # Use GPU if available
    cfg.base.cuda = torch.cuda.is_available()
    if cfg.base.cuda:
        cfg.base.num_gpu = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
    # Main function
    main(cfg)