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

# from torchpack.utils.config import configs
# from mmcv.cnn import fuse_conv_bn
# from mmcv import Config
# from mmdet3d.models import build_model
# from mmdet3d.utils import recursive_eval
# from mmcv.runner import wrap_fp16_model
import torch
import argparse
import os

# custom functional package
import lean.funcs as funcs
import lean.exptool as exptool
import lean.quantize as quantize 



from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.models import build_network
from pcdet.utils import common_utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    parser.add_argument("--in-channel", type=int, default=4, help="SCN num of input channels")
    # parser.add_argument('--ckpt', type=str, default="/data_ssd/zyy/OpenPCDet/qat/ckpt/bevfusion_ptq.pth")
    parser.add_argument('--ckpt', type=str, default="/data_ssd/zyy/OpenPCDet/output/nuscenes_models/bevfusion/default/ckpt/checkpoint_epoch_1.pth")
    parser.add_argument("--save", type=str, default="qat/onnx/lidar.backbone.onnx", help="output onnx")
    parser.add_argument("--inverse", action="store_true", help="Transfer the coordinate order of the index from xyz to zyx")
    parser.add_argument('--cfg_file', type=str, default='/data_ssd/zyy/OpenPCDet/tools/cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for demo')
   
    args = parser.parse_args()
    inverse_indices = args.inverse
    if inverse_indices:
        args.save = os.path.splitext(args.save)[0] + ".zyx.onnx"
    else:
        args.save = os.path.splitext(args.save)[0] + ".xyz.onnx"
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
  
    # model = torch.load(args.ckpt).module
    # model.eval().cuda().half()
    # model = model.encoders.lidar.backbone
   
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
    model=model.backbone_3d
    model.eval().cuda().half()
    # quantize.disable_quantization(model).apply()

    # Set layer attributes
    for name, module in model.named_modules():
        module.precision = "fp16"
        module.output_precision = "fp16"
   
    model.conv_input.precision = "fp16"
    model.conv_out.output_precision = "fp16"

    voxels = torch.zeros(256, args.in_channel).half().cuda()
    coors  = torch.zeros(256, 4).int().cuda()
    batch_size = 1
    
    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, args.save)
