# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import sys
import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import lean.quantize as quantize
import lean.funcs as funcs

# from lean.train import qat_train

# from mmcv import Config
# from torchpack.environ import auto_set_run_dir, set_run_dir
# from torchpack.utils.config import configs

# from mmdet3d.datasets import build_dataset,build_dataloader
# from mmdet3d.models import build_model
# from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

#Additions
# from mmcv.runner import  load_checkpoint,save_checkpoint
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.cnn import resnet
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d




from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu
def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model_state_to_cpu(model.state_dict())
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    if optimizer==None and epoch==None and it==None:
        return {'epoch': 2, 'it': 12, 'model_state': model_state, 'optimizer_state': 0, 'version': version}
    else:
        return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):

    filename = '{}'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


# def load_model(cfg, checkpoint_path = None):
#     model = build_model(cfg.model)
#     if checkpoint_path != None:
#         checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
#     return model

def quantize_net(model):
    quantize.quantize_encoders_lidar_branch(model.backbone_3d) 
     
    quantize.replace_to_quantization_module(model.image_backbone.backbone)
    quantize.replace_to_quantization_module(model.neck)
    quantize.replace_to_quantization_module(model.fuser)
    quantize.replace_to_quantization_module(model.backbone_2d)
    quantize.quantize_decoder(model.dense_head.decoder)
    model.backbone_3d = funcs.layer_fusion_bn(model.backbone_3d)
    return model
    
def main():
    quantize.initialize()  
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", metavar="FILE", default="bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml", help="config file")
    parser.add_argument("--ckpt", default="/data_ssd/zyy/OpenPCDet/output/nuscenes_models/bevfusion/default/ckpt/checkpoint_epoch_1.pth", help="the checkpoint file to resume from")
    parser.add_argument("--calibrate_batch", type=int, default=23, help="calibrate batch")
    parser.add_argument('--cfg_file', type=str, default='/data_ssd/zyy/OpenPCDet/tools/cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for demo')
    args = parser.parse_args()

    # args.ptq_only = True
    # # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)
    

    save_path = '/data_ssd/zyy/OpenPCDet/qat/ckpt/bevfusion_ptq.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # # set random seeds
    # if cfg.seed is not None:
    #     print(
    #         f"Set random seed to {cfg.seed}, "
    #         f"deterministic mode: {cfg.deterministic}"
    #     )
    #     random.seed(cfg.seed)
    #     np.random.seed(cfg.seed)
    #     torch.manual_seed(cfg.seed)
    #     if cfg.deterministic:
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False

    # dataset_train  = build_dataset(cfg.data.train)
    # dataset_test   = build_dataset(cfg.data.test)
    # print('train nums:{} val nums:{}'.format(len(dataset_train), len(dataset_test)))   
    # distributed =False
    # data_loader_train =  build_dataloader(
    #         dataset_train,
    #         samples_per_gpu=1,  
    #         workers_per_gpu=1,  
    #         dist=distributed,
    #         seed=cfg.seed,
    #     )
    # # print('DataLoad Info:', data_loader_train.batch_size, data_loader_train.num_workers)

    # #Create Model
    # model = load_model(cfg, checkpoint_path = args.ckpt)
    # model = quantize_net(model)
    # model = fuse_conv_bn(model)
    # # model = MMDataParallel(model, device_ids=[0])
    # model.eval()

    cfg_from_yaml_file(args.cfg_file, cfg)



    logger = common_utils.create_logger()
    train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False, workers=0,
    logger=logger,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=None,
    seed=None
    )

   
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
   
    model = quantize_net(model)
    model = fuse_conv_bn(model)
    ##Calibrate

    print("🔥 start calibrate 🔥 ")
    quantize.set_quantizer_fast(model)
    quantize.calibrate_model(model, train_loader, 0, None, args.calibrate_batch)
    
    quantize.disable_quantization(model.backbone_3d).apply()
    # quantize.disable_quantization(model.module.decoder.neck.deblocks[0][0]).apply()
    quantize.print_quantizer_status(model)
    

    model.backbone_3d= funcs.fuse_relu_only(model.backbone_3d)
    
    save_checkpoint(checkpoint_state(model), filename=save_path)
    print(f"Done due to ptq only! Save checkpoint to {save_path} 🤗")
    return

if __name__ == "__main__":
    main()