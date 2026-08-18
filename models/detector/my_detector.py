"""
自定义检测器: SmallDefectBackbone + TwoScaleDetect
不使用Neck，直接2尺度输出。
"""

import sys
import logging
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

from models.backbone.small_defect_backbone import SmallDefectBackbone
from models.head.two_scale_head import TwoScaleDetect
from utils.autoanchor import check_anchor_order
from utils.torch_utils import initialize_weights
from models.loss.loss import ComputeLoss

LOGGER = logging.getLogger(__name__)


class MyDetector(nn.Module):
    """
    自定义检测器: 使用 SmallDefectBackbone + TwoScaleDetect
    
    结构:
        输入 → SmallDefectBackbone → [S_small, S_large, (loss_ha)] → TwoScaleDetect → 输出
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = SmallDefectBackbone(cfg)
        self.head = TwoScaleDetect(cfg)
        
        self.names = cfg.Dataset.names
        self.inplace = cfg.Model.inplace
        self.check_head()
        
        initialize_weights(self)
        self.info()
        LOGGER.info('')
    
    def check_head(self):
        """计算stride并初始化anchor"""
        m = self.head
        s = 256
        m.inplace = self.inplace
        # forward一次以计算stride
        dummy = torch.zeros(1, self.cfg.Model.ch, s, s)
        feats = self.backbone(dummy)
        m.stride = torch.Tensor([s / x.shape[-2] for x in feats])
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride
        m.initialize_biases()
    
    def forward(self, x, gt_boxes=None, targets=None):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            gt_boxes: (B, N_i, 4) 或 None, 用于loss_ha计算
            targets: (N, 6) 用于检测损失计算
        
        Returns:
            训练: [loss_det + loss_ha, loss_dict]
            推理: 检测结果
        """
        feats = self.backbone(x, gt_boxes)
        
        if self.training and gt_boxes is not None:
            # 训练: 解包特征+loss_ha
            feats_list = feats[:2]  # [S_small, S_large]
            loss_ha = feats[2] if len(feats) > 2 else torch.tensor(0.0, device=x.device)
            
            # 检测头
            head_out = self.head(feats_list)
            
            # 计算检测损失
            if targets is not None:
                loss_fn = ComputeLoss(self, self.cfg)
                loss_det, loss_dict = loss_fn(head_out, targets)
                loss_dict['loss_ha'] = loss_ha.detach()
                loss_total = loss_det + loss_ha
                loss_dict['loss'] = loss_total
                return loss_total, loss_dict
            else:
                return head_out
        else:
            # 推理: 只用前两个特征
            return self.head(feats[:2])
    
    def fuse(self):
        """融合Conv+BN层"""
        LOGGER.info('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, 'bn'):
                # 简单融合（如有需要可以增强）
                pass
        return self
    
    def info(self, verbose=False, img_size=640):
        """打印模型信息"""
        model_info(self, verbose, img_size)


def model_info(model, verbose=False, img_size=640):
    """模型信息"""
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p:,} parameters, {n_g:,} gradients")