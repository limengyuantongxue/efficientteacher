"""
2尺度检测头 (Two-Scale Detection Head)
用于配合 SmallDefectBackbone 输出的小尺度(stride=4)和大尺度(stride=32)特征。
"""

import torch
import torch.nn as nn
import math


class TwoScaleDetect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, cfg):
        super().__init__()
        self.nc = cfg.Dataset.nc
        self.num_keypoints = cfg.Dataset.np if hasattr(cfg.Dataset, 'np') else 0
        self.cur_imgsize = [cfg.Dataset.img_size, cfg.Dataset.img_size]
        
        anchors = cfg.Model.anchors
        # 从cfg获取输入通道数（不使用Neck，直接取backbone输出通道）
        # small_output_channels, large_output_channels
        ch = [cfg.Model.Backbone.small_channels, cfg.Model.Backbone.large_channels]
        
        self.no = self.nc + self.num_keypoints + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.stride = cfg.Model.Head.strides
        self.export = False
    
    def initialize_biases(self, cf=None):
        for mi, s in zip(self.m, self.stride):
            b = mi.bias.view(self.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []
        list_x = [feat for feat in x[:self.nl]]
        x = list_x
        class_range = list(range(5 + self.nc))
        
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            
            if hasattr(self, 'export') and self.export:
                x[i] = x[i].view(bs, self.na, self.no, -1).permute(0, 1, 3, 2)
                z.append(x[i])
                continue
            
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                
                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        
        if hasattr(self, 'export') and self.export:
            return z
        
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid

    def post_process(self, x):
        z = []
        for i in range(self.nl):
            ny = int(self.cur_imgsize[0] / self.stride[i])
            nx = int(self.cur_imgsize[1] / self.stride[i])
            bs, _, nyxnx, _ = x[i].shape
            x[i] = x[i].view(bs, self.na, ny, nx, self.no).contiguous()
            
            if not self.training:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                y = torch.full_like(x[i], 0)
                self.anchor_grid[i] = self.anchor_grid[i].to(x[i].device)
                class_range = list(range(5 + self.nc))
                y[..., class_range] = x[i][..., class_range].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else [torch.cat(z, 1), x]