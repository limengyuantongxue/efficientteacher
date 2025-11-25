# EfficientTeacher by Alibaba Cloud 
"""
Self supervised utils
"""

import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
import random
from utils.general import clip_coords, xyxy2xywh, xywh2xyxy, xywhn2xyxy, non_max_suppression, box_iou
from utils.general import non_max_suppression_ssod
from utils.plots import plot_images_ssod, plot_images, plot_labels,  output_to_target_ssod
from utils.torch_utils import time_sync
from utils.self_supervised_utils import online_label_transform
import copy
import matplotlib.pyplot as plt
import sklearn.mixture as skm

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output

class LabelMatch(nn.Module):
    def __init__(self, cfg, target_data_len, label_num_per_img, cls_ratio_gt):
        super().__init__()
        self.nc = cls_ratio_gt.shape[0]  # 目标检测类别数
        self.multi_label = cfg.SSOD.multi_label  # 是否为多标签检测
        self.nms_conf_thres = cfg.SSOD.nms_conf_thres  # NMS置信度阈值
        self.nms_iou_thres = cfg.SSOD.nms_iou_thres  # NMS IOU阈值
        #  每个类别的高低置信度筛选阈值
        self.cls_thr_high = [cfg.SSOD.ignore_thres_high] * self.nc
        self.cls_thr_low = [cfg.SSOD.ignore_thres_low] * self.nc
        # self.percent = 0.4
        self.max_cls_per_img = 10  # 每张图片最多保存的伪标签数
        # self.queue_num = 128 * 100
        self.boxes_per_image_gt = 7  #
        self.queue_num = 128
        self.queue_len = self.queue_num * self.max_cls_per_img  # 分数队列总长度
        # self.queue_ptr = torch.zeros(1, dtype=torch.long)
        # self.cls_ratio_gt = np.array([1.0/self.nc] * self.nc)
        self.cls_ratio_gt = cls_ratio_gt  # 真实数据的类别分布比例（用于平衡伪标签的类别分布）
        self.register_buffer("score_queue" , torch.zeros(self.nc, self.queue_len))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.score_list = [[] for _ in range(self.nc)]  # 临时存储每个类别的预测分数
        self.cls_list = [range(self.nc)]
        self.start_update = False  # 标记队列是否填满，是否开始动态更新阈值
        # 伪标签最低、最高置信度阈值
        self.ignore_thres_low = cfg.SSOD.ignore_thres_low
        self.ignore_thres_high = cfg.SSOD.ignore_thres_high

        self.score_list_epoch = [[] for _ in range(self.nc)]  # 每个epoch存储的类别分数（用于更新阈值）
        self.debug = cfg.SSOD.debug
        self.resample_high_percent = cfg.SSOD.resample_high_percent  # 高分伪标签的采样比例
        self.resample_low_percent = cfg.SSOD.resample_low_percent   # 低分伪标签的采样比例
        self.names = cfg.Dataset.names  # 类别名称（用于可视化）
        self.num_points = cfg.Dataset.np  # 关键点数量（若检测任务含关键点）
        self.target_data_len = target_data_len  # 无标签数据的总长度
        self.anno_num_per_img = label_num_per_img * 3  # 每张图的标注数量上限
        # self.pos_location_low = target_data_len * label_num_per_img * self.cls_ratio_gt * 3
        # self.pos_location_high = target_data_len * label_num_per_img * self.cls_ratio_gt * self.percent * 3
        self.cls_num_list = [0 for _ in range(self.nc)]  # 计数器：统计真实标签的类别分布、有标签/无标签图像数
        self.count = 0  # 有标签图像的数量
        self.pse_count = 0  # 无标签图像的数量

        self.cls_tmp = np.zeros(self.nc)  # 临时统计真实标签的类别计数
        self.cls_num_total = np.zeros(self.nc)  # 每个类别累计的伪标签数量
        # print('total pseudo label numb:', target_data_len * label_num_per_img, 'pos_high:', self.pos_location_high)
        # print(self.pos_location_low)
    
    def _dequeue_and_enqueue(self, score):
        """
        update score queue
        将教师模型的预测分数存入队列，持续存储教师模型的预测分数，为后续统计和阈值更新提供数据
        """
        print('check device:', score[0, :])
        if dist.is_initialized():  # 分布式训练
            score = concat_all_gather(score, 1) 
        print('result:', score[0, :])
        batch_size = score.shape[1]
        ptr = int(self.queue_ptr)  # 当前队列的存储指针
        assert self.queue_len % batch_size == 0  # 确保队列长度是批次的整数倍（避免存储溢出）
        if ptr + batch_size > self.queue_len:
            lens = self.queue_len - ptr
            self.score_queue[:, ptr:ptr + lens] = score[:, :lens]
        else:
            self.score_queue[:, ptr:ptr + batch_size] = score
        #  队列首次填满，标记开始动态更新阈值
        if (not self.start_update) and ptr + batch_size >= self.queue_len:
            self.start_update = True
        ptr = (ptr + batch_size) % self.queue_len
        self.queue_ptr[0] = ptr
    
    # def store_epoch_score(self, score):
    #     for i, s in enumerate(score):
    #         s = s.cpu().numpy()
    #         s = s[s > 0]
    #         if s.shape[0] > 0:
    #             self.score_list_epoch[i].extend(s)
        # self.score_list_epoch = [np.concatenate(c) for c in self.score_list_epoch]
    def update(self, labels, n=1, pse_n=1):
        """
        统计真实标签的类别分布
        Args:
            labels:
            n:
            pse_n:

        Returns:

        """
        # self.val = val
        self.count += n  # 累计有标签图像数
        self.pse_count += pse_n  # 累计无标签图像数
        for l in labels:
            # for l in label:
                # self.cls_tmp[int(l[0:1])] += 1
            # print(l)
            self.cls_tmp[int(l[1:2])] += 1
        # label_num_per_img = np.sum(self.cls_tmp) / self.count
        # print('label_num_per_img:', label_num_per_img)
    
    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.教师模型的预测分数列表
            given_gt_thr: 默认阈值（当分数数量不足时使用）
            policy: 阈值选择策略（'high'：选高分簇的最低分；'middle'：选高分簇的中位数）

        Returns:
            float: Found gt_thr.自适应确定的伪标签筛选阈值

        """
        if len(scores) < 4:  # 分数数量不足，使用默认阈值
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        # 初始化GMM的参数：2个高斯簇（分别对应高质量/低质量伪标签）
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]  # 精度矩阵，协方差矩阵的逆
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)  # 每个分数所属的簇， 每个分数被分配到第几个高斯分量，完成聚类
        gmm_scores = gmm.score_samples(scores)  # 每个分数的似然度，评估样本与该GMM模型的匹配程度
        assert policy in ['middle', 'high']  # 断言语句，验证某个条件必须成立，如果不成立会抛出AssertionError异常
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf  # 屏蔽低质量簇的似然度，只保留高质量簇
                indx = np.argmax(gmm_scores, axis=0)  # 找到高质量簇中似然度最高的分数索引
                # 筛选出高质量簇中≥该分数的所有样本，取其最小值作为阈值（保证高分簇的完整性）
                pos_indx = (gmm_assignment == 1) & (
                        scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                pos_thr = max(given_gt_thr, pos_thr)  # 不低于默认值
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr
    
    def update_epoch_cls_thr(self, epoch):
        """
        该方法在每个 Epoch 结束时执行，结合统计分数分布和GMM 自适应阈值，更新每个类别的cls_thr_high（高分阈值）和cls_thr_low（低分阈值），为下一个 Epoch 的伪标签生成提供筛选依据
        Args:
            epoch:

        Returns:

        """
        # self.cls_ratio_gt = self.cls_tmp/np.sum(self.cls_tmp)
        # label_num_per_img = np.sum(self.cls_tmp) / self.count
        # print('label_num_per_img:', label_num_per_img)
        print('cls_tmp:', self.cls_ratio_gt)
        print('count:', self.count)
        print('target_data_len:', self.pse_count)  # 无标签图像数
        # self.pos_location_low = self.pse_count * self.cls_tmp / self.count
        # self.pos_location_high = self.pos_location_low * self.percent 
        # print('pos_location_low:', self.pos_location_low)
        # print('pos_location_high:', self.pos_location_high)
        # total_bbox_num = 0
        # for c_list in self.score_list_epoch:
            # total_bbox_num += len(c_list)
            # if len(c_list) == 0:
                # c_list.append(self.ignore_thres_low)

        for cls in range(self.nc):  # 遍历每个类别
            single_cls_score = self.score_list_epoch[cls]
            single_cls_score.sort(reverse=True)
            self.cls_num_total[cls] += len(single_cls_score)  # 累计该类别伪标签数量
            max_pseudo_label_num = int(self.cls_num_total[cls] / (epoch + 1))  # 该类别伪标签的数量上限
            # plt.figure()
            # plt.plot(single_cls_score)
            # plt.savefig(str(cls) + '.jpg')
            if len(single_cls_score) == 0:
                self.cls_thr_high[cls] = self.ignore_thres_high
                self.cls_thr_low[cls] = self.ignore_thres_low
                pos_loc_high = -1
                pos_loc_low = -1
            else:
                pos_loc_high = int(len(single_cls_score) * self.resample_high_percent)
                pos_loc_low = min(max_pseudo_label_num, int(len(single_cls_score) * self.resample_low_percent))
                # pos_loc_high = min(len(single_cls_score) - 1, int(self.pos_location_high[cls]))
                # pos_loc_low = min(len(single_cls_score) - 1, int(self.pos_location_low[cls]))
            # self.cls_thr[cls] = max(self.ignore_thres, single_cls_score[pos_loc])
            # self.cls_thr_high[cls] = max(self.ignore_thres_high, single_cls_score[pos_loc_high])
                # self.cls_thr_high[cls] = max(self.ignore_thres_high, single_cls_score[pos_loc_high])
                # GMM确定高分阈值
                gmm_thr = self.gmm_policy(np.array(single_cls_score), given_gt_thr=0.0, policy='high')
                # self.cls_thr_high[cls] = self.ignore_thres_high
                self.cls_thr_high[cls] = gmm_thr
                # print(single_cls_score[pos_loc_high], '|', gmm_thr)
                # self.cls_thr_high[cls] = single_cls_score[pos_loc_high]
                self.cls_thr_low[cls] = max(self.ignore_thres_low, single_cls_score[pos_loc_low])
                # self.cls_thr_low[cls] = self.ignore_thres_low
            LOGGER.info(f'{len(single_cls_score)} {max_pseudo_label_num} {pos_loc_high}:{self.cls_thr_high[cls]} {pos_loc_low}:{self.cls_thr_low[cls]}')
        # 重置状态，为下epoch做准备
        self.score_list_epoch = [[] for _ in range(self.nc)]
        self.cls_tmp = np.zeros(self.nc)
        self.count = 0
        self.pse_count = 0

    
    # def update_cls_thr_old(self):
    #     for cls in range(self.nc):
    #         if len(self.score_list[cls]) >= 1:
    #             self.score_list[cls].sort()
    #             cur = int(len(self.score_list[cls])/2)
    #             s = self.score_list[cls][cur]
    #         else:
    #             s = 0.1
    #         self.cls_thr[cls] = max(0.1, s)
    #     info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(self.cls_thr)])
    #     LOGGER.info(f'update score thr (positive): {info}')
    
    # def update_cls_thr(self):
    #     #等待队列收满才开始更新得分阈值
    #     if self.start_update:
    #         score, _ = self.score_queue.sort(dim=0, descending=True)
    #         boxes_per_img = self.boxes_per_image_gt * self.percent
    #         pos_location = self.queue_num * boxes_per_img * self.cls_ratio_gt
    #         # pos_location = [0] * self.nc
    #         # print('pos_location:', pos_location)
    #         for cls in range(self.nc):
    #             LOGGER.info(f'{pos_location[cls]}:{score[cls, int(pos_location[cls])]}')
    #             self.cls_thr[cls] = max(self.ignore_thres, score[cls, int(pos_location[cls])].item())
    #         # info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(self.cls_thr)])
    #         info = ' '.join([f'({v:.2f}-{i})' for i, v in enumerate(score[:, 1])])
    #         LOGGER.info(f'update score thr (positive): {info}')
    
    # def clean_score_list(self):
        # self.score_list = [[] for _ in range(self.nc)]
    
    def create_pseudo_label_online_with_gt(self, out, target_imgs, M_s, target_imgs_ori, gt=None, RANK=-2):
        """
        伪标签生成主入口，从教师模型的预测结果触发，经过NMS 过滤、分数收集、在线数据增强、坐标归一化等步骤，生成最终的伪标签，并标记其有效性
        Args:
            out:
            target_imgs:
            M_s:
            target_imgs_ori:
            gt:
            RANK:

        Returns:

        """
        n_img, _, height, width = target_imgs.shape  # batch size, channels, height, width
        lb = []
      
        # 利用非极大值抑制在线计算可靠的过滤阈值，NMS过滤教师模型的预测结果（去除重复框、低置信度框）
        pseudo_out = non_max_suppression_ssod(out, conf_thres=self.nms_conf_thres, iou_thres=self.nms_iou_thres,\
             labels=lb, num_points=self.num_points, multi_label=self.multi_label)

        refine_out = []
        score_list_batch = [[] for _ in range(self.nc)]  # 临时存储本批次的类别分数
        # 步骤2：遍历NMS后的预测结果，收集分数并格式化伪标签，分数是目标置信度
        for i, o in enumerate(pseudo_out):
            for *box, conf, cls, obj_conf, cls_conf in o.cpu().numpy():
                score_list_batch[int(cls)].append(float(conf))  # 收集该类别置信度
                # if conf > self.cls_thr_low[int(cls)]:
                # 将预测框转换为「图像索引、类别、xywh坐标、置信度」的格式，存入refine_out
                refine_out.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, obj_conf, cls_conf])
                # refine_out.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
                # if conf > self.ignore_thres_low:
                self.score_list_epoch[int(cls)].append(float(conf))

        # 步骤3：分数队列的格式化（适配分布式存储），将不同图像的类别得分列表统一为固定长度
        score_list = [[] for _ in range(self.nc)]
        for c in range(len(score_list_batch)):
                if len(score_list_batch[c]) < self.max_cls_per_img:
                    # 用0填充得分列表至长度self.max_cls_per_img
                    score_list_batch[c].extend([0.0] * (self.max_cls_per_img - len(score_list_batch[c])))
                    score_list_batch[c].sort(reverse=True)  # 从大到小排序，保留高分在前
                else:
                    score_list_batch[c].sort(reverse=True)
                    score_list_batch[c] = score_list_batch[c][:self.max_cls_per_img]
        try:
            score = np.array(score_list_batch).astype(np.float32)
            score = torch.from_numpy(score).contiguous()
        except:
            for fp in score_list:
                print(len(fp))
                if len(fp) != 160:
                    print(fp)

        # 步骤4：伪标签的格式转换与有效性初始化
        refine_out = np.array(refine_out)
        target_out_targets = torch.tensor(refine_out)  # 将伪标签转换为tensor
        target_shape = target_out_targets.shape
        target_out_targets_perspective = []  # 伪标签列表
        invalid_target_shape = True  # 标记伪标签是否有效，初始为无效
        # 无伪标签，直接返回
        if target_shape[0] == 0:
            return target_out_targets_perspective, invalid_target_shape

        # 步骤5：在线数据增强（透视变换）+ 坐标归一化
        if(target_shape[0] > 0 and target_shape[1] > 6):
            # 遍历原始无标签图像
            for i, img in enumerate(target_imgs_ori):
                image_targets = refine_out[refine_out[:, 0] == i]  # 返回第i个图像的伪标签，格式为[i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, obj_conf, cls_conf]
                # 确保伪标签为numpy数组
                if isinstance(image_targets, torch.Tensor):
                    image_targets = image_targets.cpu().numpy()

                image_targets[:, 2:6] = xywh2xyxy(image_targets[:, 2:6])  # 将锚框转换为xyxyx格式
                # 读取透视变换矩阵M（用于数据增强）
                M_select = M_s[M_s[:, 0] == i, :]  # image targets
                M = M_select[0][1:10].reshape([3,3]).cpu().numpy()
                s = float(M_select[0][10])  # 缩放因子
                ud = int(M_select[0][11])  # 上下翻转标记
                lr = int(M_select[0][12])  # 左右翻转标记
                # img, image_targets_random = self.online_label_transform_with_image(img, copy.deepcopy(image_targets[:, 1:]),  M, s, ud, lr)
                # 对图像和伪标签执行透视变换（数据增强）
                img, image_targets_random = online_label_transform(img, copy.deepcopy(image_targets[:, 1:]),  M, s)
                # img_list.append(img)

                image_targets = np.array(image_targets_random)
                if image_targets.shape[0] != 0:
                    # 为image_target的每一行的最后一列增加一列序号i
                    image_targets = np.concatenate((np.array(np.ones([image_targets.shape[0], 1]) * i), np.array(image_targets)), 1)
                    image_targets[:, 2:6] = xyxy2xywh(image_targets[:, 2:6])  # convert xyxy to xywh
                    image_targets[:, [3, 5]] /= height # normalized height 0-1
                    image_targets[:, [2, 4]] /= width # normalized width 0-1
                    image_targets[:, 2:6] = image_targets[:, 2:6].clip(0, 1)
                    # 反转标签
                    if ud == 1:
                        image_targets[:, 3] = 1 - image_targets[:, 3]
                    if lr == 1:
                        image_targets[:, 2] = 1 - image_targets[:, 2]
                    target_out_targets_perspective.extend(image_targets.tolist())
            target_out_targets_perspective = torch.from_numpy(np.array(target_out_targets_perspective))
        # img_list = torch.stack(img_list, 0)
        # if self.RANK in [-1, 0]:
            # print('total time cost:', time_sync() - total_t1)
        # 步骤6：标记伪标签为有效，并开启调试可视化
        if target_shape[0] > 0 and len(target_out_targets_perspective) > 0 :
            invalid_target_shape = False
            if self.debug and RANK in [-1 ,0]:
                draw_image = plot_images_ssod(copy.deepcopy(target_imgs), target_out_targets_perspective, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_pseudo_label_ea.jpg', names=self.names)            
                draw_image = plot_images(copy.deepcopy(target_imgs), gt, fname='/mnt/bowen/EfficientTeacher/effcient_teacher_gt_ea.jpg', names=self.names)            
        return target_out_targets_perspective, invalid_target_shape
        