"""
Scale-Aware Optimal Transport Assigner (SA-OTA)

继承 scale_aware_label_assignment.py 中的核心计算逻辑，
适配 EfficientTeacher YOLO 框架的标签分配接口。

输出格式与 YOLOAnchorAssigner.build_ota_targets 一致，
可直接被 ComputeLoss.ota_loss 使用。
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import wh_iou
from collections import OrderedDict

from utils.general import xywh2xyxy, xyxy2xywh, box_iou


def compute_w_squared(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算简化版 Wasserstein Distance 的平方（水平框版本）

    W² = 中心点距离 + 尺寸差异

    Args:
        box1: (N, 4) 或 (4,) 张量，格式为 (cx, cy, w, h)
        box2: (M, 4) 或 (4,) 张量，格式为 (cx, cy, w, h)

    Returns:
        W² 距离，标量或 (N, M) 矩阵
    """
    # 确保 box1 是 2D 的
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)

    # 提取参数
    cx1, cy1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    cx2, cy2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 中心点距离 (N, M) 或 (N,)
    if box1.shape[0] != box2.shape[0]:
        # 广播计算
        center_dist = (cx1.unsqueeze(1) - cx2.unsqueeze(0)) ** 2 + \
                      (cy1.unsqueeze(1) - cy2.unsqueeze(0)) ** 2
    else:
        center_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # 尺寸差异
    size_diff = ((w1.unsqueeze(1) if box1.shape[0] != box2.shape[0] else w1) -
                 (w2.unsqueeze(0) if box1.shape[0] != box2.shape[0] else w2)) ** 2 / 4 + \
                ((h1.unsqueeze(1) if box1.shape[0] != box2.shape[0] else h1) -
                 (h2.unsqueeze(0) if box1.shape[0] != box2.shape[0] else h2)) ** 2 / 4

    if box1.shape[0] != box2.shape[0]:
        size_diff = ((w1.unsqueeze(1) - w2.unsqueeze(0)) ** 2 +
                     (h1.unsqueeze(1) - h2.unsqueeze(0)) ** 2) / 4

    return center_dist + size_diff


def compute_nwd_similarity(boxes1: torch.Tensor, boxes2: torch.Tensor,
                           C: float = 1.0) -> torch.Tensor:
    """
    计算 NWD 相似度矩阵

    NWD(box1, box2) = exp(-W² / C)

    Args:
        boxes1: (N, 4) 张量，格式为 (cx, cy, w, h)
        boxes2: (M, 4) 张量，格式为 (cx, cy, w, h)
        C: 归一化常数，默认 1.0

    Returns:
        (N, M) 相似度矩阵，值域 [0, 1]，值越大表示越相似
               当两个框完全相同时，相似度为 1
    """
    w_squared = compute_w_squared(boxes1, boxes2)
    # 使用 1 - exp(-W²/C) 作为相似度，这样相同框的相似度为 1
    nwd = torch.exp(-w_squared / C)
    return nwd


def compute_scale_alpha(areas: torch.Tensor,
                        threshold: float = 1024.0,
                        temperature: float = 1.0) -> torch.Tensor:
    """
    尺度感知的α值计算

    公式：α = sigmoid((log(area) - log(threshold)) / temperature)

    - 面积越大，α越接近 1（主要依赖 IoU）
    - 面积越小，α越接近 0（主要依赖 NWD）
    """
    log_area = torch.log(areas.float() + 1e-8)
    log_threshold = torch.log(torch.tensor(threshold, device=areas.device))
    scale_score = (log_area - log_threshold) / temperature
    alpha = torch.sigmoid(scale_score)
    alpha = alpha.clamp(0.1, 0.9)
    return alpha


class ScaleAwareAssigner(nn.Module):
    """
    尺度感知标签分配器

    与 YOLOAnchorAssigner 兼容，输出格式与 build_ota_targets 一致。
    """
    def __init__(self,
                 nc=80,
                 na=3,
                 nl=3,
                 anchors=[],
                 anchor_t=10.0,
                 stride=None,
                 top_k=13,
                 iou_weight=3.0,
                 cls_weight=1.0,
                 scale_threshold=1024.0,
                 scale_temperature=1.0):
        super().__init__()
        self.nc = nc
        self.na = na
        self.nl = nl
        self.anchors = anchors
        self.anchor_t = anchor_t
        self.stride = stride
        self.top_k = top_k
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.scale_threshold = scale_threshold
        self.scale_temperature = scale_temperature
        self._debug_data = []
        self._collect_debug = False
        self.iou_t = 0.50


    def enable_debug_collect(self):
        self._collect_debug = True

    def disable_debug_collect(self):
        self._collect_debug = False

    def get_debug_data(self):
        return list(self._debug_data)

    def clear_debug_data(self):
        self._debug_data.clear()

    @torch.no_grad()
    def forward(self, p, targets, img_size=640):
        """
        Args:
            p: list of layer predictions [(B, na, ny, nx, no), ...]
            targets: (N, 6) [batch_idx, cls, x, y, w, h] 归一化坐标
            img_size: 输入图像尺寸（用于解码）

        Returns: 与 YOLOAnchorAssigner.build_ota_targets 输出一致
            matching_bs: [nl], 每层匹配的 batch 索引
            matching_as: [nl], 每层匹配的 anchor 索引
            matching_gjs: [nl], 每层匹配的 grid y
            matching_gis: [nl], 每层匹配的 grid x
            matching_targets: [nl], 每层匹配的目标 [batch_idx, cls, x, y, w, h]
            matching_anchs: [nl], 每层匹配的 anchor 尺寸
        """
        # Step 1: 先用 find_3_positive 筛出候选正样本
        indices, anch = self._find_3_positive(p, targets)

        matching_bs = [[] for _ in range(self.nl)]
        matching_as = [[] for _ in range(self.nl)]
        matching_gjs = [[] for _ in range(self.nl)]
        matching_gis = [[] for _ in range(self.nl)]
        matching_targets = [[] for _ in range(self.nl)]
        matching_anchs = [[] for _ in range(self.nl)]

        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            # 解码 GT 框到绝对坐标
            txywh = this_target[:, 2:6] * img_size
            txyxy = xywh2xyxy(txywh)

            # 收集所有层的候选预测
            pxyxys, p_cls, p_obj, from_which_layer = [], [], [], []
            all_b, all_a, all_gj, all_gi, all_anch = [], [], [], [], []

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(len(b), device=b.device) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:5+self.nc])

                grid = torch.stack([gi, gj], dim=1).float()
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            if not pxyxys:
                continue
            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            # Step 2: 计算尺度感知的 IoU + NWD 混合成本
            pair_wise_iou = box_iou(txyxy, pxyxys)

            # 计算 GT 面积
            gt_areas = this_target[:, 4] * this_target[:, 5] * img_size * img_size
            alpha = compute_scale_alpha(gt_areas, self.scale_threshold, self.scale_temperature)

            # NWD 需要 (cx, cy, w, h) 格式
            # txywh 已经是 (cx, cy, w, h) 格式
            # pxyxys 是 (x1,y1,x2,y2) 格式，需要转成 (cx,cy,w,h)
            pxywh_for_nwd = xyxy2xywh(pxyxys)
            nwd = compute_nwd_similarity(txywh, pxywh_for_nwd, C=50.0)

            # 混合相似度 = α·IoU + (1-α)·NWD
            mixed_sim = alpha.unsqueeze(1) * pair_wise_iou + (1 - alpha.unsqueeze(1)) * nwd
            pair_wise_mixed_loss = -torch.log(mixed_sim + 1e-8)

            # 动态 K — 与 OTA 一致
            top_k = min(self.top_k, pair_wise_mixed_loss.shape[1])
            topk_mixedsim, _ = torch.topk(mixed_sim, top_k, dim=1)
            dynamic_ks = torch.clamp(topk_mixedsim.sum(1).int(), min=1)

            topk_iou, _ = torch.topk(pair_wise_iou, top_k, dim=1)
            dynamic_ks_iou = torch.clamp(topk_iou.sum(1).int(), min=1)

            if self._collect_debug:
                gt_areas_cpu = gt_areas.detach().cpu()
                dk_cpu = dynamic_ks.detach().cpu()
                dk_iou_cpu = dynamic_ks_iou.detach().cpu()
                alpha_cpu = alpha.detach().cpu()
                basic_cnt = self._count_basic_positives(p, this_target)
                basic_cpu = basic_cnt.detach().cpu()
                for gi in range(this_target.shape[0]):
                    self._debug_data.append(OrderedDict([
                        ('area', round(float(gt_areas_cpu[gi]), 2)),
                        ('dynamic_ks_iou', int(dk_iou_cpu[gi])),
                        ('dynamic_ks', int(dk_cpu[gi])),
                        ('basic_pos_count', int(basic_cpu[gi])),
                        ('alpha', round(float(alpha_cpu[gi]), 4)),
                    ]))

            # 分类成本 — 使用
            # Cost_cls = CE(P, Y_soft) × (Y_soft − P)²
            # P = p_obj（objectness logits）
            # Y_soft = IoU(gt_boxes, pred_boxes) 作为软标签
            num_gt = this_target.shape[0]
            num_pred = pxyxys.shape[0]

            # 用mixed_sim作为软标签 Y_soft
            # gt 框用 txyxy (x1y1x2y2), 预测框用 pxyxys (x1y1x2y2)
            y_soft = mixed_sim.detach()

            # 对每个 GT 计算分类成本
            pair_wise_cls_loss = torch.zeros((num_gt, num_pred), device=p_obj.device)
            # 用 p_obj 作为 pred_scores（logits 形式）
            pred_score = p_obj # (N, 1) objectness logits
            # 交叉熵: BCEWithLogits(P, Y_soft)
            ce = F.binary_cross_entropy_with_logits(
                pred_score.T.repeat(num_gt,1),
                y_soft,  # Y_soft 扩展维度作为目标
                reduction='none'
            )  # (N,)

            # 预测概率
            prob = pred_score.sigmoid()[:, 0]  # (N,)

            # 调制因子 (Y_soft - P)²
            modulation = (y_soft - prob) ** 2

            pair_wise_cls_loss = ce * modulation

            # 总成本
            cost = self.cls_weight * pair_wise_cls_loss + self.iou_weight * pair_wise_mixed_loss

            # Step 3: sa 匹配
            matching_matrix = torch.zeros_like(cost)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0

            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]
            this_target = this_target[matched_gt_inds]

            for i in range(self.nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        # 转换为每层 Tensor
        for i in range(self.nl):
            if matching_targets[i]:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                device = p[0].device
                matching_bs[i] = torch.tensor([], device=device, dtype=torch.int64)
                matching_as[i] = torch.tensor([], device=device, dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device=device, dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device=device, dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device=device, dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device=device, dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs
        #  matching_bs, matching_as, matching_gjs, matching_gis分别对应incides中的b,a,gj,gi，matching_targets是归一化完整的GT,形状 [n, 6]: [img_id, cls, x, y, w, h]，matching_anchs是对应的anchor尺寸，形状 [n, 2]: [w, h]

    def _count_basic_positives(self, p, targets):
        na, nt = self.na, targets.shape[0]
        device = targets.device

        # Track original GT index through the filtering chain
        gt_idx = torch.arange(nt, device=device).view(1, nt, 1).repeat(na, 1, 1).float()
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)
        targets_cat = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None], gt_idx), 2)
        # shape: [na, nt, 8] -> [img, cls, x, y, w, h, anchor_idx, gt_idx]

        gain = torch.ones(8, device=device)
        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=device).float() * g
        pos_counts = torch.zeros(nt, dtype=torch.int32, device=device)

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            t = targets_cat * gain
            if nt == 0:
                continue
            r = t[:, :, 4:6] / anchors_i[:, None]
            # j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t  # 基于长宽比匹配
            j = wh_iou(anchors_i, t[:, :,4:6]) >self.iou_t  # 基于iou静态匹配
            t = t[j]
            if t.shape[0] == 0:
                continue
            gxy = t[:, 2:4]
            gxi = gain[[2, 3]] - gxy
            jk = ((gxy % 1. < g) & (gxy > 1.)).T
            lm = ((gxi % 1. < g) & (gxi > 1.)).T
            mask = torch.stack((torch.ones_like(jk[0]), jk[0], jk[1], lm[0], lm[1]))
            t = t.repeat((5, 1, 1))[mask]
            if t.shape[0] == 0:
                continue
            gids = t[:, 7].long()
            for gid in gids.unique():
                pos_counts[gid] += (gids == gid).sum()

        return pos_counts

    def _find_3_positive(self, p, targets):
        """与 YOLOAnchorAssigner.find_3_positive 相同"""
        na, nt = self.na, targets.shape[0]
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets_cat = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1]],
                           device=targets.device).float() * g

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
            t = targets_cat * gain
            if nt:
                r = t[:, :, 4:6] / anchors_i[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_t
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                jk = ((gxy % 1. < g) & (gxy > 1.)).T
                lm = ((gxi % 1. < g) & (gxi > 1.)).T
                mask = torch.stack((torch.ones_like(jk[0]), jk[0], jk[1], lm[0], lm[1]))
                t = t.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
            else:
                t = targets_cat[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            anch.append(anchors_i[a])

        return indices, anch