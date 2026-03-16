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


import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify
# from torchpack.utils.config import configs
# from mmcv import Config
# from mmdet3d.models import build_model
# from mmdet3d.utils import recursive_eval

from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
# import lean.quantize as quantize

from pcdet.datasets import build_dataloader
from pcdet.config import cfg, cfg_from_yaml_file

from pcdet.models import build_network
from pcdet.utils import common_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Export bevfusion model")
    parser.add_argument('--ckpt', type=str, default="/data_ssd/zyy/OpenPCDet/output/nuscenes_models/bevfusion/default/ckpt/checkpoint_epoch_1.pth")
    # parser.add_argument('--ckpt', type=str, default="/data_ssd/zyy/OpenPCDet/qat/ckpt/bevfusion_ptq.pth")
    parser.add_argument('--fp16', action= 'store_true')
    parser.add_argument('--cfg_file', type=str, default='/data_ssd/zyy/OpenPCDet/tools/cfgs/nuscenes_models/bevfusion.yaml',
                        help='specify the config for demo')
    args = parser.parse_args()
    return args


class SubclassDownSample(nn.Module):
    def __init__(self,model):
        super(SubclassDownSample, self).__init__()
        self.downsample=model.vtransform.downsample
    def forward(self,feat):
        feat = self.downsample(feat)
        return feat

class SubclassCameraModule(nn.Module):
    def __init__(self, model):
        super(SubclassCameraModule, self).__init__()
        self.backbone = model.image_backbone.backbone
        self.neck = model.neck
        self.vtransform=model.vtransform
    def forward(self, img, depth):
        # breakpoint()
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)
        
       
        x = self.backbone.conv1(img)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        c2 = self.backbone.layer1(x)   # 256
        c3 = self.backbone.layer2(c2)  # 512
        c4 = self.backbone.layer3(c3)  # 1024
        c5 = self.backbone.layer4(c4)  # 2048
        
        
        batch_dict = {'image_features': [c3, c4, c5]}
        neck_output = self.neck(batch_dict)
      

        def get_cam_feats(self, x, d):
            # B, N, C, fH, fW = map(int, x.shape)
            d = d.view(B * N, *d.shape[2:])
            # x = x.view(B * N, C, fH, fW)

            d = self.dtransform(d)
            x = torch.cat([d, x], dim=1)
            x = self.depthnet(x)

            depth = x[:, : self.D].softmax(dim=1)
            feat  = x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1)
            return feat, depth
        
        return get_cam_feats(self.vtransform, neck_output['image_fpn'][0], depth)

def main():
    args = parse_args()
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
    suffix = "fp16"
    # if args.fp16:
    #     suffix = "fp16"
    #     quantize.disable_quantization(model).apply()
        
    B, N, C, H, W = 1, 4, 3, 480, 480
    img = torch.zeros(B, N, C, H, W).cuda().half()
    # points = [i.cuda() for i in data["points"].data[0]]

    camera_model = SubclassCameraModule(model)
    camera_model.cuda().eval().half()
    depth = torch.zeros(B, N, 1, H, W).cuda().half()
    for name, module in model.named_modules():
        if hasattr(module, '_input_quantizer'):
            module._input_quantizer._fake_quant = True
            module._input_quantizer._learn_amax = False
            print(f"Configured quantizer for {name}")
    downsample_model = SubclassDownSample(model)
    downsample_model.cuda().eval()
    downsample_in = torch.zeros(1, 80, 360, 360).cuda().half()

    save_root = f"qat/onnx_{suffix}"
    os.makedirs(save_root, exist_ok=True)
    
    with torch.no_grad():
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"
        camera_vtransform_onnx = f"{save_root}/camera.vtransform.onnx"
        TensorQuantizer.use_fb_fake_quant = True
        torch.onnx.export(
            camera_model,
            (img, depth),
            camera_backbone_onnx,
            input_names=["img", "depth"],
            output_names=["camera_feature", "camera_depth_weights"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig = onnx.load(camera_backbone_onnx)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_simp, camera_backbone_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")

        torch.onnx.export(
            downsample_model,
            downsample_in,
            camera_vtransform_onnx,
            input_names=["feat_in"],
            output_names=["feat_out"],
            opset_version=13,
            do_constant_folding=True,
        )

        onnx_orig_downsample = onnx.load(camera_vtransform_onnx)
        onnx_simp_downsample, check_downsample = simplify(onnx_orig_downsample)
        onnx.save(onnx_simp_downsample, camera_vtransform_onnx)
        print(f"🚀 The export is completed. ONNX save as {camera_vtransform_onnx} 🤗, Have a nice day~")
        model1 = onnx.load(f"{save_root}/camera.backbone.onnx")
        model2 = onnx.load(f"{save_root}/camera.vtransform.onnx")
        try:
            onnx.checker.check_model(model1)
            onnx.checker.check_model(model2)
            print(" ONNX model1 is valid")
        except Exception as e:
            print(f" ONNX model is invalid: {e}")

        # has_quant = any(node.op_type in ["QuantizeLinear", "DequantizeLinear"] 
        #             for node in model1.graph.node)
    
        # if has_quant:
        #     print("✅  is QUANTIZED")
        # else:
        #     print(" is NOT quantized")


        # for init in model1.graph.initializer:
        #     if init.name.endswith('.weight'):
        #         data_type = init.data_type
        #         print(f"Weight {init.name}: type {data_type} (1=float32, 3=int8, ...)")
        #         break


if __name__ == "__main__":
    main()
