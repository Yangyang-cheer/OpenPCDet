import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights


class ResNet50(nn.Module):

    def __init__(self,model_cfg, pretrained=True):

        super(ResNet50, self).__init__()
        
        # 加载ResNet50模型
        if pretrained:
            # 使用官方最新的预训练权重
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            print("Loaded ImageNet pretrained weights (V2)")
        else:
            self.backbone = models.resnet50(weights=None)
            print("Initialized with random weights")
        
        # # 提取各个组件
        # self.conv1 = self.backbone.conv1
        # self.bn1 = self.backbone.bn1
        # self.relu = self.backbone.relu
        # self.maxpool = self.backbone.maxpool
        self.frozen_stages = model_cfg.FROZEN_STAGE
        self.output_stride = model_cfg.OUTPUT_STRIDE
        
        # # 四个残差阶段
        # self.layer1 = self.backbone.layer1  
        # self.layer2 = self.backbone.layer2  
        # self.layer3 = self.backbone.layer3  
        # self.layer4 = self.backbone.layer4  
        

        
        # 如果output_stride=16，修改layer4的步长
        if self.output_stride == 16:
            for module in self.layer4.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)
            
            if hasattr(self.layer4[0], 'downsample') and self.layer4[0].downsample is not None:
                for m in self.layer4[0].downsample.modules():
                    if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                        m.stride = (1, 1)
        
        # 冻结指定层
        
        self._freeze_stages()
    
    def _freeze_stages(self):
        """冻结指定阶段的参数"""
        if self.frozen_stages >= 0:
            # 冻结stem
            self.backbone.conv1.eval()
            self.backbone.bn1.eval()
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 1:
            # 冻结layer1
            self.backbone.layer1.eval()
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 2:
            # 冻结layer2
            self.backbone.layer2.eval()
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 3:
            # 冻结layer3
            self.layer3.eval()
            for param in self.backbone.layer3.parameters():
                param.requires_grad = False
        
        if self.frozen_stages >= 4:
            # 冻结layer4
            self.backbone.layer4.eval()
            for param in self.backbone.layer4.parameters():
                param.requires_grad = False

        

    
    def forward(self, batch_dict):
        # Stem
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        outs=[]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # Stage 1
        x = self.backbone.layer1(x)
        # features['layer1'] = x  # 步长4
       
        
        # Stage 2
        x = self.backbone.layer2(x)
        outs.append(x)
        
        # Stage 3
        x = self.backbone.layer3(x)
        outs.append(x)
        
        # Stage 4
        x = self.backbone.layer4(x)
        outs.append(x)
        batch_dict['image_features'] = outs
       
        return batch_dict
    
