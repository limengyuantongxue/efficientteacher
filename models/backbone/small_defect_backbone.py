"""
SmallDefectBackbone —— 把小缺陷专用特征提取网络(SDFEN)包装成 YOLO 的 backbone。

设计要点：
  1. 只替换 backbone；输出 3 路特征(stride 8/16/32)，直接接原 PANet neck + Detect head。
  2. 输出通道与 YOLOv5m(width_multiple=0.75) 对齐为 192/384/768，因此 neck/head 配置无需改动。
  3. 同时把各层激活图暴露为 self._ha_maps，供 ComputeLoss 计算 HA 损失。
"""

import torch
import torch.nn as nn

from .small_defect_network import SmallDefectNetwork, _HaMapsStore


class SmallDefectBackbone(nn.Module):
    """SDFEN backbone 包装器。

    前向返回 (S_small, S_medium, S_large)，分辨率分别为 H/8, H/16, H/32；
    并设置 self._ha_maps = [M1, M2, M3, M4]（各层尺度激活图，3 通道）。
    """

    def __init__(self, cfg):
        super().__init__()
        bc = cfg.Model.Backbone
        self.net = SmallDefectNetwork(
            in_channels=cfg.Model.ch,
            widths=list(getattr(bc, 'widths', [64, 128, 256, 512])),
            boundary_ratio=float(getattr(bc, 'boundary_ratio', 0.001)),
            boundary_ratio_large=float(getattr(bc, 'boundary_ratio_large', 0.002)),
            out_channels_small=int(getattr(bc, 'small_channels', 192)),
            out_channels_medium=int(getattr(bc, 'medium_channels', 384)),
            out_channels_large=int(getattr(bc, 'large_channels', 768)),
            ha_temperature=float(getattr(bc, 'ha_temperature', 1.0)),
        )
        # 供 ComputeLoss 读取；非 SDFEN backbone 没有这个属性
        # 用安全容器保存，避免 EMA deepcopy / checkpoint pickle 非叶子张量
        self._ha_store = _HaMapsStore(None)

    # ---- _ha_maps: 对外接口保持为 list[Tensor]（ComputeLoss 直接 len/slice）----
    @property
    def _ha_maps(self):
        return self._ha_store.maps

    @_ha_maps.setter
    def _ha_maps(self, value):
        self._ha_store.maps = value

    def compute_ha_loss(self, maps, gt_xyxy, img_size=640):
        """层次激活损失（供 ComputeLoss 调用）；gt_xyxy: (B, N_i, 4) 归一化 xyxy，无效行 -1。"""
        return self.net._compute_ha_loss(maps, gt_xyxy, img_size)

    def forward(self, x):
        feats = self.net(x)                    # (S_small, S_medium, S_large)
        self._ha_maps = self.net.last_maps     # 各层激活图
        return feats
