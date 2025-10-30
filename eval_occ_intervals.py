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

    occ_intervals = mng.cfg.data.get('occ_intervals', [0, 0.25, 0.5, 0.75, 1])  # the larger, the more visible, less occluded.
    print(occ_intervals)

    occ_metric_lst = []
    for i in range(len(occ_intervals)):  # for each interval, maintain a list, plus (1, )
        occ_metric_lst.append([[],[],[],[]])  # joints_metric, joints_metric_aligned, verts_metric, verts_metric_aligned

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
                else:
                    batch_size = mng.cfg.test.batch_size
                    
                if "DEX" in mng.cfg.data.name:
                    batch_input = tool.tensor_gpu(batch_input, check_on=False)
                    batch_input = [{k: v[bid] for k, v in batch_input.items()} for bid in range(batch_size)]
                    
                    batch_output = tool.tensor_gpu(batch_output, check_on=False)
                    batch_output = [{k: v[bid] for k, v in batch_output.items()} for bid in range(batch_size)]
                    
                    pred_lst = mng.dataset[split].get_predictions(batch_input, batch_output)
                    for b in range(batch_size):  # for different sample
                        sample_occ_ratio = batch_input[b]['gt_occ_ratio']

                        for interval_idx in range(len(occ_intervals)-1):  # for each interval
                            occ_flag = sample_occ_ratio >= occ_intervals[interval_idx] and sample_occ_ratio <= occ_intervals[interval_idx+1]
                            
                            if occ_flag:
                                occ_metric_lst[interval_idx][0].append(pred_lst[0][b])
                                occ_metric_lst[interval_idx][1].append(pred_lst[1][b])
                                occ_metric_lst[interval_idx][2].append(pred_lst[2][b])
                                occ_metric_lst[interval_idx][3].append(pred_lst[3][b])

                                # make sure not fall into other intervals
                                break

                        if not occ_flag:  # if not fall into any interval, it should be larger than 1
                                occ_metric_lst[-1][0].append(pred_lst[0][b])
                                occ_metric_lst[-1][1].append(pred_lst[1][b])
                                occ_metric_lst[-1][2].append(pred_lst[2][b])
                                occ_metric_lst[-1][3].append(pred_lst[3][b])

                # Tqdm settings
                t.set_description(desc="")
                t.update()
                
            t.close()


    for i in range(len(occ_metric_lst)-1):
        interval_left = occ_intervals[i]
        interval_right = occ_intervals[i+1]
        occ_metric = occ_metric_lst[i]

        print("[{}, {}]: {}".format(interval_left, interval_right, len(occ_metric[0])))

        # import ipdb; ipdb.set_trace()
        gt_xyz_list = np.stack(occ_metric[2], axis=0)
        gt_verts_list = np.stack(occ_metric[3], axis=0)
        pred_xyz_list = np.stack(occ_metric[0], axis=0)
        pred_verts_list = np.stack(occ_metric[1], axis=0)

        output_dir_occ = osp.join(cfg.base.model_dir, 'occ_[{},{}]'.format(interval_left, interval_right))
        os.makedirs(output_dir_occ, exist_ok=True)
        eval.main(gt_xyz_list, gt_verts_list, pred_xyz_list, pred_verts_list, output_dir_occ)

    
    occ_metric = occ_metric_lst[-1]
    print("(1, ): {}".format(len(occ_metric[0])))
    
    if len(occ_metric[0]) > 0:
        gt_xyz_list = np.stack(occ_metric[2], axis=0)
        gt_verts_list = np.stack(occ_metric[3], axis=0)
        pred_xyz_list = np.stack(occ_metric[0], axis=0)
        pred_verts_list = np.stack(occ_metric[1], axis=0)

        output_dir_occ = osp.join(cfg.base.model_dir, 'occ_(1,)')
        os.makedirs(output_dir_occ, exist_ok=True)
        eval.main(gt_xyz_list, gt_verts_list, pred_xyz_list, pred_verts_list, output_dir_occ)


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