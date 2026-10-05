# -*- coding: utf-8 -*-
"""
双重自适应伪标签阈值（Dual-Adaptive Pseudo-Label Threshold, DAPT）
=====================================================================
移植并扩展自 code/dapt_threshold.py（原版，2026-06）。

相对原版的扩展（本次集成所需）：
  1. 双阈值来源（方案 a：只自适应 high，low 固定）：
         high = T = mu_low + alpha_t * (mu_high - mu_low)   ← 原脚本/论文的融合公式（偏向高分）
         low  = 管线固定低阈值（ignore_thres_low，默认 0.05），DAPT 不改动它
     动机：低阈值是【丢弃线】。若让 DAPT 抬高它，会大量丢弃本可作为软监督的伪标签，
           导致半监督监督不足（实测 mAP50 -14.2 点）。故 DAPT 只决定【多高才算可靠(硬标签)】，
           不改变【用不用】这个总量，只改变【硬/软监督的配比】。
  2. 新增 on_epoch_end() / get_thresholds()：每 batch 累积、每 epoch 重算，
     并按【样本数】从固定阈值渐进混合到 DAPT 阈值（冷启动保护）。
  3. 支持 dapt_mode ∈ {class, scale, dual} 的消融（仅组键不同，代码只有一套）。
  4. 原版的 compute_thresholds() / get_threshold() / filter_pseudo_labels() 全部保留，
     便于将来做"单阈值变体"的消融。

工作流：
    update()          每 batch 累积伪标签分数（按 类别×尺度 分组）
    on_epoch_end()    每 epoch 重算各组阈值（BGMM 两分量均值 + 动量平滑 + 样本数混合）
    get_thresholds()  查询某条伪标签对应的 (high, low) 阈值
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

try:
    from sklearn.mixture import BayesianGaussianMixture
except ImportError:  # pragma: no cover
    BayesianGaussianMixture = None


class DAPTThresholdManager:
    """双重自适应伪标签阈值管理器。

    Args:
        mode:             'dual' (类别×尺度) | 'class' (仅类别) | 'scale' (仅尺度)
        estimator:        'bgmm' | 'percentile'
        scale_threshold:  小/大目标的【面积占比】阈值 theta_s（与 SaA 的尺度阈值同口径）
        fit_min:          允许拟合 BGMM 的最小样本数（低于此值直接用固定阈值）
        min_samples:      完全信任 DAPT 所需的样本数（达到前按样本数渐进混合）
        max_buffer_size:  每组缓冲区上限（滑动窗口）
        momentum:         阈值跨 epoch 的动量平滑系数 tau
        percentile:       分位数估计器使用：high 取该分位，low 取 (100-该值) 分位
        logger:           可选，logging.Logger
    """

    def __init__(self,
                 mode: str = 'dual',
                 estimator: str = 'bgmm',
                 scale_threshold: float = 0.001,
                 fit_min: int = 20,
                 min_samples: int = 200,
                 max_buffer_size: int = 2000,
                 momentum: float = 0.9,
                 percentile: float = 65,
                 alpha_t: float = 0.6,
                 T_floor: float = 0.15,
                 target_rate=0.65,
                 logger=None):
        assert mode in ('class', 'scale', 'dual'), f'不支持的 dapt_mode: {mode}'
        assert estimator in ('bgmm', 'percentile', 'quantile'), f'不支持的 dapt_estimator: {estimator}'
        self.mode = mode
        self.estimator = estimator
        self.scale_threshold = float(scale_threshold)
        self.fit_min = int(fit_min)
        self.min_samples = int(min_samples)
        self.max_buffer_size = int(max_buffer_size)
        self.momentum = float(momentum)
        self.percentile = float(percentile)
        self.alpha_t = float(alpha_t)   # 融合公式 T = mu_low + alpha_t*(mu_high-mu_low) 中的 alpha_t
        # T 的下界：必须低于固定 high(默认 0.30)，否则 DAPT 只能单向变严、无法给小目标组放松
        self.T_floor = float(T_floor)
        # 分位数匹配的目标保留率 rho：
        #   数值(推荐) -> 固定 rho：目标不动，不与模型形成反馈环；每组恰好保留 rho 比例
        #   'auto'     -> 每轮用固定阈值在缓冲区上重算 r*（旧行为，会与模型互相驱动而震荡）
        if isinstance(target_rate, str) and target_rate.lower() == 'auto':
            self.target_rate = 'auto'
        else:
            self.target_rate = float(target_rate)
        self.logger = logger

        # 分组缓冲：{key: [scores]}
        self.buffer = defaultdict(list)
        # 当前生效的 (high, low)：{key: (high, low)}
        self.cur_hl: Dict[tuple, Tuple[float, float]] = {}
        # 动量平滑后的 DAPT 估计（混合前）：{key: (mu_high, mu_low)}
        self.ema_hl: Dict[tuple, Tuple[float, float]] = {}
        self.last_epoch_stats: List[dict] = []
        self.last_r_star = None      # quantile 估计器：上一轮算出的全局基线保留率 r*

    # ------------------------------------------------------------------ #
    # 分组
    # ------------------------------------------------------------------ #
    def get_scale(self, area_ratio: float) -> str:
        """按面积占比判定尺度类别。"""
        return 'small' if area_ratio < self.scale_threshold else 'large'

    def key_for(self, cls_idx: int, area_ratio: float) -> tuple:
        """按 dapt_mode 生成组键（三种消融共用同一套代码）。"""
        scale = self.get_scale(area_ratio)
        if self.mode == 'dual':
            return (int(cls_idx), scale)
        if self.mode == 'class':
            return (int(cls_idx), 'all')
        return ('all', scale)  # scale-only

    # ------------------------------------------------------------------ #
    # 累积
    # ------------------------------------------------------------------ #
    def update(self, cls_idx: int, confidence: float, area_ratio: float):
        """累积一条伪标签分数（每个 batch 调用）。"""
        key = self.key_for(cls_idx, area_ratio)
        buf = self.buffer[key]
        buf.append(float(confidence))
        if len(buf) > self.max_buffer_size:
            del buf[:len(buf) - self.max_buffer_size]

    def update_records(self, records: List[dict]):
        """批量累积。records: [{'cls_idx','confidence','area_ratio'}, ...]"""
        for r in records:
            self.update(r['cls_idx'], r['confidence'], r['area_ratio'])

    # ------------------------------------------------------------------ #
    # 估计器
    # ------------------------------------------------------------------ #
    def _fit_bgmm_means(self, scores: List[float]):
        """BGMM 双分量拟合，返回 (mu_low, mu_high)；失败返回 None。"""
        if BayesianGaussianMixture is None or len(scores) < self.fit_min:
            return None
        try:
            X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
            bgmm = BayesianGaussianMixture(
                n_components=2,
                covariance_type='full',
                max_iter=100,
                random_state=42,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.01,
            )
            bgmm.fit(X)
            means = bgmm.means_.flatten()
            high_idx = int(np.argmax(means))
            low_idx = 1 - high_idx
            return float(means[low_idx]), float(means[high_idx])
        except Exception as e:
            if self.logger:
                self.logger.warning(f'[DAPT] BGMM 拟合失败: {e}')
            return None

    def _fit_percentile_means(self, scores: List[float]):
        """分位数备选：low 取 (100-p) 分位，high 取 p 分位。"""
        if len(scores) < self.fit_min:
            return None
        lo = float(np.percentile(scores, 100.0 - self.percentile))
        hi = float(np.percentile(scores, self.percentile))
        return lo, hi

    # ------------------------------------------------------------------ #
    # 每 epoch 重算
    # ------------------------------------------------------------------ #
    def _compute_baseline_keep_rate(self, fh_arr):
        """分位数匹配用：计算【当前固定阈值下】的全局保留率 r*（按样本数加权）。

        含义：如果用固定 high 去筛，全体候选里有多大比例能被保留。
        后续让【每一个组】都按这个比例保留 —— 这样：
          * 总体保留量 = 固定阈值下的总体保留量（总量不变）
          * 组与组之间不再出现"有的组留 80%、有的组只剩 25%"的不成比例丢弃
        """
        num = den = 0.0
        for key, scores in self.buffer.items():
            n = len(scores)
            if n == 0:
                continue
            cls_idx = key[0] if isinstance(key[0], (int, np.integer)) else None
            fh = float(fh_arr[cls_idx]) if (cls_idx is not None and cls_idx < len(fh_arr)) else float(fh_arr.mean())
            keep = float(np.mean(np.asarray(scores, dtype=np.float64) >= fh))
            num += n * keep
            den += n
        if den <= 0:
            return None
        # 夹到 [0.05, 0.95]，避免分位数退化到极值
        return float(np.clip(num / den, 0.05, 0.95))

    def on_epoch_end(self, fixed_high, fixed_low):
        """重算各组阈值。

        Args:
            fixed_high / fixed_low: 按类别索引的固定阈值序列（回退与混合的基准）
        """
        fh_arr = np.asarray(fixed_high, dtype=np.float64).reshape(-1)
        fl_arr = np.asarray(fixed_low, dtype=np.float64).reshape(-1)
        stats = []

        # quantile 估计器：确定【所有组共用的目标保留率】
        if self.estimator == 'quantile':
            if self.target_rate == 'auto':
                # 旧行为：用固定阈值在缓冲区上实测 r*（自指 → 与模型形成反馈环，实测会剧烈震荡）
                r_star = self._compute_baseline_keep_rate(fh_arr)
            else:
                # 推荐：固定 rho —— 目标不随模型漂移，反馈环被切断
                r_star = float(self.target_rate)
        else:
            r_star = None
        self.last_r_star = r_star

        for key, scores in self.buffer.items():
            n = len(scores)
            cls_idx = key[0] if isinstance(key[0], (int, np.integer)) else None
            fh = float(fh_arr[cls_idx]) if cls_idx is not None and cls_idx < len(fh_arr) else float(fh_arr.mean())
            fl = float(fl_arr[cls_idx]) if cls_idx is not None and cls_idx < len(fl_arr) else float(fl_arr.mean())

            # 样本太少 → 直接用固定阈值
            if n < self.fit_min:
                self.cur_hl[key] = (fh, fl)
                stats.append(dict(key=key, n=n, high=fh, low=fl, w=0.0, src='fixed'))
                continue

            # ================= 分支 1：分位数匹配（新方法 c2） =================
            if self.estimator == 'quantile':
                if r_star is None:
                    self.cur_hl[key] = (fh, fl)
                    stats.append(dict(key=key, n=n, high=fh, low=fl, w=0.0, r=r_star, src='fixed(no-r*)'))
                    continue
                # 取组内 (1 - r*) 分位数 → 该组恰好保留 r* 比例的伪标签
                t_raw = float(np.quantile(np.asarray(scores, dtype=np.float64), 1.0 - r_star))
                # 仅做很宽的数值保护（正常不会触发）
                t_raw = float(np.clip(t_raw, 0.02, 0.98))
                src = 'quantile'
                # 说明：分位数本身已由滑动窗口平滑，且为了严格保持"总量不变"，
                #       此分支【不加动量】。

            # ================= 分支 2：BGMM / 分位数均值（原方法，完整保留） =================
            else:
                means = (self._fit_bgmm_means(scores) if self.estimator == 'bgmm'
                         else self._fit_percentile_means(scores))

                if means is None:                  # 拟合失败 → 回退固定阈值
                    self.cur_hl[key] = (fh, fl)
                    stats.append(dict(key=key, n=n, high=fh, low=fl, w=0.0, r=r_star, src='fixed(fit-fail)'))
                    continue

                mu_low_raw, mu_high_raw = means
                # 论文/原脚本公式：T = mu_low + alpha_t*(mu_high - mu_low)，偏向高分
                t_raw = mu_low_raw + self.alpha_t * (mu_high_raw - mu_low_raw)
                # T 的下界可配置（默认 0.15 < 固定 high 0.30）
                t_raw = float(np.clip(t_raw, self.T_floor, 0.95))
                src = 'dapt'

                # 动量平滑（只作用于 T）
                if key in self.ema_hl:
                    t_raw = self.momentum * self.ema_hl[key] + (1.0 - self.momentum) * t_raw
                self.ema_hl[key] = t_raw

            # 按样本数从固定阈值渐进混合到 DAPT 阈值
            w = min(1.0, n / float(self.min_samples))
            high = (1.0 - w) * fh + w * t_raw
            low = fl        # 方案 a：只自适应 high，low 固定为管线固定低阈值
            self.cur_hl[key] = (high, low)
            stats.append(dict(key=key, n=n, high=high, low=low, w=w, r=r_star, src=src))

        self.last_epoch_stats = stats
        return stats

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_thresholds(self, cls_idx: int, area_ratio: float,
                       fixed_high: float, fixed_low: float) -> Tuple[float, float]:
        """返回该条伪标签对应的 (high, low)。无 DAPT 估计时回退固定阈值。"""
        key = self.key_for(cls_idx, area_ratio)
        if key in self.cur_hl:
            return self.cur_hl[key]
        return float(fixed_high), float(fixed_low)

    # ------------------------------------------------------------------ #
    # 诊断
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict:
        return {str(k): dict(n=len(v),
                             mean=float(np.mean(v)) if v else 0.0,
                             std=float(np.std(v)) if v else 0.0,
                             min=float(np.min(v)) if v else 0.0,
                             max=float(np.max(v)) if v else 0.0)
                for k, v in self.buffer.items()}

    def reset(self):
        self.buffer.clear()
        self.cur_hl.clear()
        self.ema_hl.clear()
        self.last_epoch_stats = []
        self.last_r_star = None