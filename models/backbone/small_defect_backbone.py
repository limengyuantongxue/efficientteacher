import torch
import torch.nn as nn
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from code.small_defect_network import SmallDefectNetwork


class SmallDefectBackbone(nn.Module):
    """
    包装 SmallDefectNetwork 作为YOLO的backbone。
    
    训练时: 返回 [S_small, S_large, loss_ha]
    推理时: 返回 [S_small, S_large]
    """
    def __init__(self, cfg):
        super().__init__()
        base_channels = cfg.Model.Backbone.base_channels
        self.net = SmallDefectNetwork(
            in_channels=cfg.Model.ch,
            base_channels=base_channels,
            boundary_ratio=cfg.Model.Backbone.boundary_ratio
        )
    
    def forward(self, x, gt_boxes=None):
        """
        Args:
            x: (B, 3, H, W) 输入图像
            gt_boxes: (B, N_i, 4) 或 None, 归一化坐标
        
        Returns:
            features: [S_small, S_large] 两个尺度的特征
            或 [S_small, S_large, loss_ha] (训练时有gt_boxes)
        """
        S_small, S_large, loss_ha = self.net(x, gt_boxes)
        
        if self.training and gt_boxes is not None:
            return [S_small, S_large, loss_ha]
        else:
            return [S_small, S_large]