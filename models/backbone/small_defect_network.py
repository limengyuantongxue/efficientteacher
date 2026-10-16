"""
小缺陷专用特征提取网络 (Small Defect Detection Network)

融合了三篇论文的核心概念：
1. HANet论文: 尺度引导特征激活(SGFA) + 尺度特定特征重耦合(SSFC)
2. Semi-SSDDet论文: 空洞卷积块(DB) + 哈尔小波下采样(HWD)
3. 用户设计: 前置卷积块 + 双尺度解耦 + SPPF

网络结构：
- 前置卷积块: CBR → HWD → CBR → HWD (2次下采样)
- 空洞卷积块1-3: CBR → DCB(d=2,4,8) → Concat → HWD
- 空洞卷积块4: CBR → DCB(d=2,4,8) → Concat → SPPF
- 双尺度特征解耦层: 对P1,P2,P3,P4分别解耦
- 特征重耦合层: 尺度对齐 + 注意力聚合

Author: 论文研究
Date: 2026-07-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ==================== 激活图暂存容器 ====================

class _HaMapsStore:
    """激活图暂存器（只用于「当前前向 -> 紧接着的 loss 计算」之间传参）。

    为什么需要它：各层激活图 M 是带 autograd 图的中间张量。若直接把它们存成
    模块属性（list[Tensor]），会带来两个致命问题：
      1) ModelEMA 构建时会 deepcopy(model)  -> 非叶子张量不支持 deepcopy；
      2) 保存 checkpoint 时会 pickle 整个 model -> 非叶子张量不支持 pickle。
    本容器把张量隔离在一层"壳"里：deepcopy / pickle 该容器时一律返回空容器，
    于是 EMA 副本与 checkpoint 都不会携带这些临时张量，而训练时仍能正常回传梯度。
    """
    __slots__ = ('maps',)

    def __init__(self, maps=None):
        self.maps = maps

    def __deepcopy__(self, memo):
        return _HaMapsStore(None)

    def __reduce__(self):
        return (_HaMapsStore, (None,))


# ==================== 基础模块 ====================

class CBR(nn.Module):
    """
    标准卷积块: Conv + BatchNorm + ReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 扩张率
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = None, dilation: int = 1):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class HWD(nn.Module):
    """
    哈尔小波下采样 (Haar Wavelet Downsampling)
    
    使用二维离散小波变换替代传统池化，保留高频细节信息。
    输出4个子带: LL(低频近似), LH(水平细节), HL(垂直细节), HH(对角细节)
    
    输入: (B, C, H, W)
    输出: (B, C, H/2, W/2)   ← 4 个子带经 1×1 卷积融合回 C 通道（通道数不变）
    
    特点:
    - 可逆操作，信息完整保留
    - 特别适合对细节敏感的微小缺陷检测
    """
    def __init__(self, channels: int):
        super().__init__()
        # 用1×1卷积将4个子带(4C)融合压缩回 C —— 信息经卷积融合保留，但通道不再膨胀
        self.compress = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        二维哈尔小波分解
        
        将2×2区域分解为4个分量：
        - LL: 低频近似 (求和)
        - LH: 水平细节 (行差)
        - HL: 垂直细节 (列差)
        - HH: 对角细节 (对角差)
        """
        # 提取2×2区域的四个位置
        x1 = x[:, :, 0::2, 0::2]  # 左上
        x2 = x[:, :, 1::2, 0::2]  # 左下
        x3 = x[:, :, 0::2, 1::2]  # 右上
        x4 = x[:, :, 1::2, 1::2]  # 右下
        
        # 哈尔小波变换
        LL = (x1 + x2 + x3 + x4) * 0.5  # 低频近似
        LH = (x1 + x2 - x3 - x4) * 0.5  # 水平细节
        HL = (x1 - x2 + x3 - x4) * 0.5  # 垂直细节
        HH = (x1 - x2 - x3 + x4) * 0.5  # 对角细节
        
        # 拼接4个子带并压缩通道
        out = torch.cat([LL, LH, HL, HH], dim=1)
        return self.compress(out)


class DCB(nn.Module):
    """
    空洞卷积块 (Dilated Convolution Block)
    
    并行使用3个不同扩张率的空洞卷积，扩大感受野而不增加参数。
    扩张率: d=2(感受野5×5), d=4(感受野9×9), d=8(感受野17×17)
    
    结构:
        输入 → [DConv(d=2), DConv(d=4), DConv(d=8)] → Concat → 融合卷积 → 输出
    """
    def __init__(self, channels: int):
        super().__init__()
        c = channels // 3  # 每个分支的通道数
        
        # 并行多尺度空洞卷积
        self.d2 = CBR(channels, c, kernel_size=3, dilation=2)
        self.d4 = CBR(channels, c, kernel_size=3, dilation=4)
        self.d8 = CBR(channels, c, kernel_size=3, dilation=8)
        
        # 融合层
        self.fuse = CBR(c * 3, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行处理
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        
        # 拼接融合
        out = torch.cat([d2, d4, d8], dim=1)
        return self.fuse(out)


class SPPF(nn.Module):
    """
    空间金字塔池化 (Spatial Pyramid Pooling Fast)
    
    YOLOv5/v8中常用的SPPF变体，通过级联最大池化获取多尺度上下文。
    仅在第4个空洞卷积块后使用，不改变空间尺寸。
    """
    def __init__(self, channels: int, pool_size: int = 5):
        super().__init__()
        c = channels // 2
        self.cv1 = CBR(channels, c, kernel_size=1)
        self.cv2 = CBR(c * 4, channels, kernel_size=1)
        self.m = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ==================== 双尺度特征解耦层 ====================

class ScaleDecouple(nn.Module):
    """
    双尺度特征解耦层 (Dual-Scale Feature Decoupling)
    
    受HANet论文的SGFA启发，学习一个激活图将特征分解为三个尺度子空间（小/中/大）。
    
    结构:
        输入特征 → 卷积 + Softmax → 激活图M(3通道: small/medium/large)
        输出: {小尺度子空间, 大尺度子空间}
    
    特点:
    - 仅激活前景区域，抑制背景干扰
    - 软标签避免硬划分边界
    """
    def __init__(self, channels: int):
        super().__init__()
        # 学习尺度激活图
        self.activate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: 输入特征 (B, C, H, W)

        Returns:
            feat_small  : 小尺度子空间   (B, C, H, W)
            feat_medium : 中尺度子空间   (B, C, H, W)
            feat_large  : 大尺度子空间   (B, C, H, W)
            M           : 激活图         (B, 3, H, W)  通道顺序 = [small, medium, large]
        """
        M = self.activate(feat)  # (B, 3, H, W)

        feat_small  = feat * M[:, 0:1, :, :].expand_as(feat)
        feat_medium = feat * M[:, 1:2, :, :].expand_as(feat)
        feat_large  = feat * M[:, 2:3, :, :].expand_as(feat)

        return feat_small, feat_medium, feat_large, M


# ==================== 特征重耦合层 ====================

class FeatureReCouple(nn.Module):
    """
    尺度特定特征重耦合 (Scale-Specific Feature Re-Coupling)
    
    受HANet论文的SSFC启发，将不同层级中同一尺度的特征子空间对齐并聚合。
    
    结构:
        1. 通道对齐: 各层特征用1×1卷积统一到目标通道数
        2. 尺度对齐: 各层特征上/下采样到统一尺寸
        3. 注意力聚合: Attention * Sum(features)
    
    Args:
        in_channels_list: 各层输入通道数列表
        out_channels: 输出通道数
    """
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        # 通道对齐层: 各层1×1卷积统一到out_channels
        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        # 注意力权重生成
        self.attn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, feat_list: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            feat_list: 特征列表 [P1, P2, P3, P4]
            target_size: 对齐目标尺寸 (H, W)
        
        Returns:
            增强的尺度特定特征
        """
        aligned = []
        for i, f in enumerate(feat_list):
            # 通道对齐
            f = self.align[i](f)
            # 尺度对齐
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(f)
        
        # 求和聚合
        A = sum(aligned)
        
        # 注意力增强
        return self.attn(A) * A


# ==================== 完整网络 ====================

class SmallDefectNetwork(nn.Module):
    """
    小缺陷专用特征提取网络（SDFEN）

    整体结构:
        输入 → Stem(CBR→HWD→CBR→HWD)
             → Block1(CBR→DCB)          → p1  (stride 4)
             → Block2(HWD→CBR→DCB)      → p2  (stride 8)
             → Block3(HWD→CBR→DCB)      → p3  (stride 16)
             → Block4(HWD→CBR→DCB→SPPF) → p4  (stride 32)
             → 三尺度解耦 → 特征重耦合 → 输出 3 路 (stride 8/16/32)

    Args:
        in_channels : 输入通道数（默认 3）
        widths      : 四个金字塔层级的通道宽度 [w1,w2,w3,w4]
        boundary_ratio       : b1 小/中 边界（归一化面积占比）
        boundary_ratio_large : b2 中/大 边界
        out_channels_small/medium/large : 三路输出的通道（须与 neck 输入一致）
    """

    def __init__(self, in_channels: int = 3, widths=None,
                 boundary_ratio: float = 0.001, boundary_ratio_large: float = 0.002,
                 out_channels_small: int = 192, out_channels_medium: int = 384,
                 out_channels_large: int = 768, ha_temperature: float = 1.0):
        super().__init__()
        if widths is None:
            widths = [128, 256, 512, 1024]
        assert len(widths) == 4, "widths 必须提供 4 个层级的通道数"
        w1, w2, w3, w4 = widths
        self.widths = list(widths)

        # b1: 小/中 边界；b2: 中/大 边界（均为归一化面积占比）
        self.boundary_ratio = boundary_ratio
        self.boundary_ratio_large = boundary_ratio_large
        # 软标签 sigmoid 的温度（单位：像素）。越大，边界附近的"模糊带"越宽。
        self.ha_temperature = ha_temperature

        # === 前置卷积块 (Stem): CBR → HWD → CBR → HWD ===
        self.stem = nn.Sequential(
            CBR(in_channels, w1, kernel_size=7, stride=1, padding=3),
            HWD(w1),
            CBR(w1, w1, kernel_size=3, stride=1),
            HWD(w1),
        )

        # === 空洞卷积块 1-3 ===
        self.block1 = nn.Sequential(
            CBR(w1, w1, kernel_size=3, stride=1),
            DCB(w1),
        )
        self.block2 = nn.Sequential(
            HWD(w1),
            CBR(w1, w2, kernel_size=3, stride=1),
            DCB(w2),
        )
        self.block3 = nn.Sequential(
            HWD(w2),
            CBR(w2, w3, kernel_size=3, stride=1),
            DCB(w3),
        )

        # === 空洞卷积块 4: CBR → DCB → SPPF ===
        self.block4 = nn.Sequential(
            HWD(w3),
            CBR(w3, w4, kernel_size=3, stride=1),
            DCB(w4),
            SPPF(w4),
        )

        # === 三尺度解耦层（对 p1~p4 各一个）===
        self.decouple = nn.ModuleList([
            ScaleDecouple(w1),
            ScaleDecouple(w2),
            ScaleDecouple(w3),
            ScaleDecouple(w4),
        ])

        # === 特征重耦合层：三路，输入为各层通道 widths ===
        self.recouple_small  = FeatureReCouple(list(widths), out_channels_small)
        self.recouple_medium = FeatureReCouple(list(widths), out_channels_medium)
        self.recouple_large  = FeatureReCouple(list(widths), out_channels_large)

        # 供外部（ComputeLoss）读取各层激活图（容器见 _HaMapsStore）
        self._ha_store = _HaMapsStore()

        # === 初始化权重 ===
        self._initialize_weights()

    # ---- last_maps: 激活图的对外接口；真正存放于 deepcopy/pickle 安全的容器 ----
    @property
    def last_maps(self):
        return self._ha_store.maps

    @last_maps.setter
    def last_maps(self, value):
        self._ha_store.maps = value

    def forward(self, x: torch.Tensor, gt_boxes: Optional[torch.Tensor] = None):
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W)
            gt_boxes: 仅为兼容旧调用保留；HA 损失已改由 ComputeLoss 负责，此处不再计算

        Returns:
            S_small  : 小尺度增强特征, stride 8   (B, 192, H/8 , W/8 )
            S_medium : 中尺度增强特征, stride 16  (B, 384, H/16, W/16)
            S_large  : 大尺度增强特征, stride 32  (B, 768, H/32, W/32)

        附加：各层激活图存入 self.last_maps，供 ComputeLoss 计算 HA 损失。
        """
        # === Stem阶段 ===
        x = self.stem(x)                      # H/4

        # === 4 个空洞卷积块（实际分辨率 p1=H/4, p2=H/8, p3=H/16, p4=H/32）===
        p1 = self.block1(x)                   # H/4
        p2 = self.block2(p1)                  # H/8
        p3 = self.block3(p2)                  # H/16
        p4 = self.block4(p3)                  # H/32

        # === 三尺度解耦 ===
        feats = [p1, p2, p3, p4]
        small_list, medium_list, large_list, maps = [], [], [], []
        for i, f in enumerate(feats):
            fs, fm, fl, m = self.decouple[i](f)
            small_list.append(fs)
            medium_list.append(fm)
            large_list.append(fl)
            maps.append(m)

        # === 特征重耦合：3 路，分别对齐到 stride 8 / 16 / 32 ===
        S_small  = self.recouple_small(small_list,   p2.shape[-2:])   # H/8
        S_medium = self.recouple_medium(medium_list, p3.shape[-2:])   # H/16
        S_large  = self.recouple_large(large_list,   p4.shape[-2:])   # H/32

        self.last_maps = maps   # 走 property -> 安全容器

        return S_small, S_medium, S_large

    def _generate_soft_labels(self, feat_h: int, feat_w: int, gt_boxes: torch.Tensor,
                              device: torch.device, img_size: int = 640):
        """在特征图分辨率下生成【三尺度软标签】与【前景掩码】。

        通道顺序 = [small, medium, large]，与 ScaleDecouple 的激活图一致。

        对每个目标（s 为其归一化面积占比）：
            d = 该目标到【所属档位相邻的那条边界】的距离
            U = 1 - 1/(1 + exp(-d)),   C = 1 - U
            own 通道 = C
            该最近边界所邻接的另一个子空间 = U
            其余通道 = 0                    （C + U = 1，与 Softmax 自洽）

        子空间划分：
            s <  b1          -> small
            b1 <= s < b2     -> medium
            s >= b2          -> large

        Args:
            gt_boxes: (B, N_i, 4) 归一化 xyxy，无效行用 -1 填充
        Returns:
            labels : (B, 3, H, W)
            fg_mask: (B, 1, H, W)  1=前景(框内)，0=背景
        """
        B = gt_boxes.shape[0]
        # 关键：距离必须用【等效像素边长】，否则 d 只有 1e-3 量级，
        # sigmoid 会恒等于 0.5，软标签退化成均匀分布、HA 损失完全学不到尺度信息。
        b1 = (self.boundary_ratio ** 0.5) * img_size
        b2 = (self.boundary_ratio_large ** 0.5) * img_size
        T = max(self.ha_temperature, 1e-6)

        labels = torch.zeros(B, 3, feat_h, feat_w, device=device)
        fg_mask = torch.zeros(B, 1, feat_h, feat_w, device=device)

        for b in range(B):
            boxes = gt_boxes[b]
            valid = (boxes[:, 0] >= 0).nonzero(as_tuple=True)[0]
            if valid.numel() == 0:
                continue
            for idx in valid:
                x1, y1, x2, y2 = boxes[idx].unbind()
                # s = 等效像素边长（不是面积占比）
                s = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).sqrt() * img_size

                zero = torch.zeros((), device=device)
                if s < b1:
                    k, d = 0, (b1 - s)
                elif s < b2:
                    k = 1
                    d = torch.minimum(s - b1, b2 - s)
                else:
                    k, d = 2, (s - b2)

                # U: 边界处=0.5，远离边界→0（d 以像素计，T 为温度）
                U = 1 - 1 / (1 + torch.exp(-d / T))
                C = 1 - U

                if k == 0:                       # small，最近边界 b1，邻接 medium
                    vals = (C, U, zero)
                elif k == 1:                     # medium，看离 b1 还是 b2 更近
                    if (s - b1) <= (b2 - s):
                        vals = (U, C, zero)      # 邻接 small
                    else:
                        vals = (zero, C, U)      # 邻接 large
                else:                            # large，最近边界 b2，邻接 medium
                    vals = (zero, U, C)

                fx1 = max(0, int(x1 * feat_w)); fy1 = max(0, int(y1 * feat_h))
                fx2 = min(feat_w, int(torch.ceil(x2 * feat_w)))
                fy2 = min(feat_h, int(torch.ceil(y2 * feat_h)))
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                for ch in range(3):
                    labels[b, ch, fy1:fy2, fx1:fx2] = vals[ch]
                fg_mask[b, 0, fy1:fy2, fx1:fx2] = 1.0

        return labels, fg_mask

    def _compute_ha_loss(self, maps, gt_boxes, img_size=640):
        """层次激活损失（HA loss）：前景 soft cross-entropy

            L = -Σ_j t_j · log(m_j)        仅在 fg_mask == 1 的像素上求均值

        各层分别生成对应分辨率的软标签，逐层求损失后取平均。
        """
        loss = 0.0
        eps = 1e-8
        n = 0
        for M in maps:
            B, _, H, W = M.shape
            labels, fg_mask = self._generate_soft_labels(H, W, gt_boxes, M.device, img_size)
            ce = -(labels * torch.log(M + eps)).sum(dim=1, keepdim=True)   # (B,1,H,W)
            denom = fg_mask.sum().clamp(min=1.0)
            loss = loss + (ce * fg_mask).sum() / denom
            n += 1
        return loss / max(n, 1)

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # === 尺度激活图最后 1x1 卷积：置零初始化（必须在通用 kaiming 之后执行）===
        # 目的：初始 softmax 恒为均匀(1/3) -> HA 损失从 log3≈1.099 起步，
        #       与检测损失(box+obj+cls)同量级；三个尺度通道无初始偏好。
        for d in self.decouple:
            last_conv = d.activate[-2]      # activate = [Conv, Conv, Conv, Softmax]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)


# ==================== 测试代码 ====================

def test_network():
    """测试网络前向传播和参数量"""
    print("=" * 60)
    print("小缺陷专用特征提取网络 - 测试")
    print("=" * 60)
    
    # 创建网络
    model = SmallDefectNetwork(in_channels=3, base_channels=3)
    model.train()
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 640, 640)  # 输入图像
    
    # 模拟真实框: 归一化坐标 [x1,y1,x2,y2]
    # 用-1填充无效行，形状为 [B, N_i, 4]
    gt_boxes = torch.tensor([
        [[0.1, 0.1, 0.15, 0.15],   # 图0-box0: 0.0025 > boundary 0.001
         [0.5, 0.5, 0.55, 0.55]],  # 图0-box1: 0.0025 > boundary
        [[0.2, 0.2, 0.22, 0.22],   # 图1-box0: 0.0004 < boundary
         [0.7, 0.7, 0.76, 0.76]],  # 图1-box1: 0.0036 > boundary
    ])  # (2, 2, 4)
    
    print(f"\n输入尺寸: {x.shape}")
    print(f"gt_boxes 形状: {gt_boxes.shape}")
    print(f"图0 边界框: {gt_boxes[0]}")
    print(f"图1 边界框: {gt_boxes[1]}")
    
    # 训练模式
    S_small, S_large, loss_ha = model(x, gt_boxes)
    print(f"\n训练模式:")
    print(f"  S_small: {S_small.shape}")
    print(f"  S_large: {S_large.shape}")
    print(f"  loss_ha: {loss_ha.item():.6f}")
    
    # 推理模式
    model.eval()
    with torch.no_grad():
        S_small, S_large, loss_ha = model(x)
    print(f"\n推理模式:")
    print(f"  S_small: {S_small.shape}")
    print(f"  S_large: {S_large.shape}")
    print(f"  loss_ha: {loss_ha.item():.6f}")
    
    # 测试各模块输出尺寸
    print("\n" + "-" * 40)
    print("各层输出尺寸分析:")
    print("-" * 40)
    
    x = torch.randn(1, 3, 640, 640)
    print(f"输入: {x.shape}")
    
    stem_out = model.stem(x)
    print(f"Stem输出: {stem_out.shape} (H/4)")
    
    p1 = model.block1(stem_out)
    print(f"Block1输出(P1): {p1.shape} (H/4)")
    
    p2 = model.block2(p1)
    print(f"Block2输出(P2): {p2.shape} (H/16)")
    
    p3 = model.block3(p2)
    print(f"Block3输出(P3): {p3.shape} (H/32)")
    
    p4 = model.block4(p3)
    print(f"Block4输出(P4): {p4.shape} (H/32, SPPF不改变尺寸)")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_network()