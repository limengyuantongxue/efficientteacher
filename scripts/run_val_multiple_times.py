# -*- coding: utf-8 -*-
"""
多次评估脚本：依次对多次训练得到的权重文件运行 val.py，
输出每次评估结果（整体 + 每个类别），并计算多次评估的均值(Mean)与标准差(Std)。

用法：
    1. 修改下方【用户配置区】中的 WEIGHTS_LIST（5 次训练的权重路径）等参数
    2. 运行: python scripts/run_val_multiple_times.py

说明：
    - 通过 subprocess 调用 val.py（与 scripts/run_train_multiple_times.py 风格一致），
      每次评估相互独立，单个权重评估失败不影响其他权重。
    - 从 val.py 打印的 "all" 汇总行解析整体 P / R / mAP@.5 / mAP@.5:.95，
      从各类别行解析每个类别的对应指标（数据集类别少时 val.py 会自动打印各类别行；
      也可在 VAL_ARGS 里把 verbose 设为 True 强制打印）。
    - 最后打印汇总表（整体 + 每个类别 + 面积分档的 Mean、Std、Mean±Std）。
      整体/类别写入 val_multi_results.csv；面积分档（AP_S/AP_M/AP_L）写入 val_multi_results_area.csv。
    - 面积分档来自 val.py 的 --area-bins 输出（COCO 口径 32^2/96^2、scale 口径 0.001/0.002 归一化面积占比）。
    - 标准差默认用样本标准差 ddof=1（论文/报告中常用口径）。
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import re
import sys
import subprocess
import time
from datetime import datetime

import numpy as np

# 添加项目根目录到路径
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(FILE))  # efficientteacher 目录
sys.path.insert(0, ROOT)

# ============================================================
# 用户配置区：请根据你的实际情况修改
# ============================================================

# 5 次训练得到的权重文件路径（请替换为你的实际路径）
WEIGHTS_LIST = [
    "D:\\个人文件\\文献\\毕业论文参考\\code\\efficientteacher\\exp_ssod\\exp46\\weights\\best.pt",
    "D:\\个人文件\\文献\\毕业论文参考\\code\\efficientteacher\\exp_ssod\\exp47\\weights\\best.pt",
    "D:\\个人文件\\文献\\毕业论文参考\\code\\efficientteacher\\exp_ssod\\exp48\\weights\\best.pt",
    "D:\\个人文件\\文献\\毕业论文参考\\code\\efficientteacher\\exp_ssod\\exp49\\weights\\best.pt",
    "D:\\个人文件\\文献\\毕业论文参考\\code\\efficientteacher\\exp_ssod\\exp50\\weights\\best.pt",
]

# 传给 val.py 的通用参数
#   bool=True   -> 加对应开关参数（如 --val-ssod）
#   bool=False  -> 不加该开关
#   空字符串    -> 不传该参数（val.py 用默认值）
VAL_ARGS = {
    # 数据集 yaml 路径（YOLOv5 格式，含 path/train/val/names）
    # 若想用训练时同款 val 划分，可改用 --cfg 传训练配置，见下方 cfg 说明
    'data': '',
    'task': '',          # train / val / test
    'batch-size': '',        # batch size（显存不足时调小）
    'imgsz': '',           # 推理尺寸（训练用的 img_size=1280，想对齐训练时指标可改 1280）
    'conf-thres': '',    # 置信度阈值（评估建议保持 0.001）
    'iou-thres': '',       # NMS IoU 阈值（评估建议保持 0.6）
    'device': '',           # 设备，如 '0' 或 '0,1'，留空自动选择
    'num-points': '',        # 关键点数量（断铅笔检测无关键点，填 0）
    'verbose': True,        # 打印每个类别的指标（便于本脚本解析各类别；类别多时也强制打印）
    'save-txt': '',      # 是否保存预测 txt
    'half': '',          # 是否使用 FP16 推理
    'val-ssod': True,       # SSOD(EfficientTeacher) 模型评估保持 True；若报错可改 False
    'cfg': '',              # 训练配置 yaml（走 cfg 里的数据集路径）；用 --data 时留空
    'area-bins': 'both',   # 'both' | 'coco' | 'scale' | 'none'：同时输出 COCO 与本文尺度口径的分档 AP
}

# 评估结果保存文件（每次结果 + 均值/标准差，含整体和各类别）
RESULT_FILE = os.path.join(os.path.dirname(FILE), 'val_multi_results.csv')

# 要统计的指标（对应 val.py 输出的汇总行）
METRIC_NAMES = ['P', 'R', 'mAP@.5', 'mAP@.5:.95']

# 面积分档：val.py 打印的 bin 名形如 COCO-S / COCO-M / COCO-L / scale-S / ...
AREA_BIN_RE = re.compile(r'^(COCO|scale)-[SML]$')
AREA_METRICS = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 'AR']

# 分档结果单独存一个 CSV，避免破坏主结果文件的列结构
AREA_RESULT_FILE = os.path.join(os.path.dirname(FILE), 'val_multi_results_area.csv')


def build_command(weights):
    """构造调用 val.py 的命令行。"""
    cmd = [sys.executable, '-W', 'ignore', os.path.join(ROOT, 'val.py')]
    for key, value in VAL_ARGS.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, (int, float)):
            cmd.append(f'--{key}')
            cmd.append(str(value))
        elif isinstance(value, str):
            if value != '':  # 空字符串不传
                cmd.append(f'--{key}')
                cmd.append(value)
        else:
            cmd.append(f'--{key}')
            cmd.append(str(value))
    cmd.append('--weights')
    cmd.append(weights)
    return cmd


def parse_val_output(stdout):
    """从 val.py 输出中解析整体('all')、各类别、以及面积分档 AP。

    返回:
        {
            'all':     {'P':.., 'R':.., 'mAP@.5':.., 'mAP@.5:.95':..},
            'classes': {'body_broken': {...}, ...},
            'area':    {'COCO-S': {'nGT':914, 'P':.., 'R':.., 'mAP@.5':..,
                                   'mAP@.5:.95':.., 'AR':..}, 'scale-S': {...}, ...}
        }
        未解析到 'all' 行时返回 None。

    说明：val.py 的分档行格式为
        <bin> <#GT> <P> <R> <mAP@.5> <mAP@.5:.95> <AR>
    其中 bin ∈ {COCO-S, COCO-M, COCO-L, scale-S, scale-M, scale-L}。
    必须**先**识别这些行，否则它们会被误当成"类别行"（字段位置不同，会读出错位的数值）。
    """
    if not stdout:
        return None
    result = {'all': None, 'classes': {}, 'area': {}}
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0]

        # ---- 面积分档行（优先处理）----
        if AREA_BIN_RE.match(name):
            if len(parts) >= 7:
                try:
                    result['area'][name] = {
                        'nGT': int(float(parts[1])),
                        'P': float(parts[2]),
                        'R': float(parts[3]),
                        'mAP@.5': float(parts[4]),
                        'mAP@.5:.95': float(parts[5]),
                        'AR': float(parts[6]),
                    }
                except (ValueError, IndexError):
                    pass
            continue

        # ---- 整体行 / 类别行 ----
        if len(parts) < 7:
            continue
        if name in ('Class', 'Speed:', 'Results', 'bin'):  # 表头/速度/保存信息
            continue
        try:
            metrics = {
                'P': float(parts[3]),
                'R': float(parts[4]),
                'mAP@.5': float(parts[5]),
                'mAP@.5:.95': float(parts[6]),
            }
        except (ValueError, IndexError):
            continue  # 非指标行
        if name == 'all':
            result['all'] = metrics
        else:
            result['classes'][name] = metrics
    if result['all'] is None:
        return None
    return result


def run_val_once(weights, run_number, total_runs):
    """对单个权重运行一次 val.py，返回 (success, metrics, duration, stdout)。"""
    print(f"\n{'=' * 70}")
    print(f"开始第 {run_number}/{total_runs} 次评估")
    print(f"权重: {weights}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}\n")

    # 清理 GPU 缓存
    try:
        import torch
        torch.cuda.empty_cache()
        print("GPU 缓存已清理")
    except Exception:
        pass

    start_time = time.time()
    cmd = build_command(weights)
    print(f"命令: {' '.join(cmd)}\n")

    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'  # 保证子进程输出按 UTF-8 打印/解析
    process = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
    )
    duration = time.time() - start_time
    stdout = process.stdout or ''

    if process.returncode != 0:
        print(f"[错误] val.py 退出码 {process.returncode}")
        stderr_tail = (process.stderr or '').strip().splitlines()[-30:]
        print("---- stderr 末尾 ----")
        print('\n'.join(stderr_tail) if stderr_tail else '(无)')
        print("---- stdout 末尾 ----")
        stdout_tail = stdout.strip().splitlines()[-20:]
        print('\n'.join(stdout_tail) if stdout_tail else '(无)')
        return False, None, duration, stdout

    metrics = parse_val_output(stdout)
    if metrics is not None:
        m = metrics['all']
        print(f"第 {run_number} 次评估结果(整体): "
              f"P={m['P']:.4f}  R={m['R']:.4f}  "
              f"mAP@.5={m['mAP@.5']:.4f}  mAP@.5:.95={m['mAP@.5:.95']:.4f}")
        for cls, cm in metrics['classes'].items():
            print(f"  [{cls}] P={cm['P']:.4f}  R={cm['R']:.4f}  "
                  f"mAP@.5={cm['mAP@.5']:.4f}  mAP@.5:.95={cm['mAP@.5:.95']:.4f}")
        if metrics.get('area'):
            print("  [分档 AP] " + " | ".join(
                f"{b}: mAP@.5={v['mAP@.5']:.4f}, AR={v['AR']:.4f}"
                for b, v in metrics['area'].items()))
    else:
        print("[警告] 未能从 val.py 输出中解析到 'all' 汇总行")
        stdout_tail = stdout.strip().splitlines()[-20:]
        print("---- stdout 末尾 ----")
        print('\n'.join(stdout_tail) if stdout_tail else '(无)')

    print(f"\n第 {run_number}/{total_runs} 次评估 {'成功' if metrics is not None else '失败'}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"评估时长: {duration:.2f} 秒 ({duration / 60:.2f} 分钟)")
    print(f"{'=' * 70}\n")

    return metrics is not None, metrics, duration, stdout


def print_metric_table(title, rows, means=None, stds=None):
    """打印一张指标表。rows: [{run, metrics}]，means/stds 为 numpy 数组或 None。"""
    header = f"{'Run':<6}{'P':>12}{'R':>12}{'mAP@.5':>12}{'mAP@.5:.95':>14}"
    print(f"\n{title}")
    print(header)
    print('-' * len(header))
    for r in rows:
        m = r['metrics']
        print(f"{r['run']:<6}{m['P']:>12.4f}{m['R']:>12.4f}"
              f"{m['mAP@.5']:>12.4f}{m['mAP@.5:.95']:>14.4f}")
    if means is not None:
        print('-' * len(header))
        print(f"{'Mean':<6}" + ''.join(f'{v:>12.4f}' for v in means))
        print(f"{'Std':<6}" + ''.join(f'{v:>12.4f}' for v in stds))


def print_area_table(title, bins, area_stats):
    """打印一组面积分档的汇总表：每个 bin 一行，数值为多次评估的 mean±std。

    #GT 来自 val.py 的统计（各次评估一致）；n 为实际参与统计的评估次数。
    """
    header = f"{'bin':<10}{'#GT':>7}{'n':>4}{'mAP@.5':>20}{'mAP@.5:.95':>20}{'AR':>20}"
    print(f"\n{title}")
    print(header)
    print('-' * len(header))
    for b in bins:
        st = area_stats[b]
        m, sd = st['means'], st['stds']
        c1 = f"{m[2]:.4f}±{sd[2]:.4f}"
        c2 = f"{m[3]:.4f}±{sd[3]:.4f}"
        c3 = f"{m[4]:.4f}±{sd[4]:.4f}"
        print(f"{b:<10}{st['nGT']:>7d}{st['runs']:>4d}{c1:>20}{c2:>20}{c3:>20}")
    print('-' * len(header))


def save_results(results, all_stats, class_stats):
    """把每次评估结果（整体+各类别）及均值/标准差保存到 CSV。

    all_stats:   {'means': np.array, 'stds': np.array} 或 None
    class_stats: {类别名: {'means': np.array, 'stds': np.array}}
    """
    try:
        with open(RESULT_FILE, 'w', encoding='utf-8-sig') as f:
            f.write('run,weights,class,P,R,mAP@.5,mAP@.5:.95\n')
            for r in results:
                w = r['weights'].replace(',', ';')
                if r['metrics'] is not None:
                    am = r['metrics']['all']
                    f.write(f"{r['run']},{w},all,{am['P']:.4f},{am['R']:.4f},"
                            f"{am['mAP@.5']:.4f},{am['mAP@.5:.95']:.4f}\n")
                    for cls, cm in r['metrics']['classes'].items():
                        f.write(f"{r['run']},{w},{cls},{cm['P']:.4f},{cm['R']:.4f},"
                                f"{cm['mAP@.5']:.4f},{cm['mAP@.5:.95']:.4f}\n")
                else:
                    f.write(f"{r['run']},{w},all,FAILED,FAILED,FAILED,FAILED\n")
            if all_stats is not None:
                f.write(f"mean,,all,{all_stats['means'][0]:.4f},{all_stats['means'][1]:.4f},"
                        f"{all_stats['means'][2]:.4f},{all_stats['means'][3]:.4f}\n")
                f.write(f"std,,all,{all_stats['stds'][0]:.4f},{all_stats['stds'][1]:.4f},"
                        f"{all_stats['stds'][2]:.4f},{all_stats['stds'][3]:.4f}\n")
            for cls, st in class_stats.items():
                f.write(f"mean,,{cls},{st['means'][0]:.4f},{st['means'][1]:.4f},"
                        f"{st['means'][2]:.4f},{st['means'][3]:.4f}\n")
                f.write(f"std,,{cls},{st['stds'][0]:.4f},{st['stds'][1]:.4f},"
                        f"{st['stds'][2]:.4f},{st['stds'][3]:.4f}\n")
        print(f"结果已保存到: {RESULT_FILE}")
    except Exception as e:
        print(f"[警告] 保存结果文件失败: {e}")


def save_area_results(results, area_stats):
    """把每次评估的分档结果 + mean/std 写入独立 CSV（不破坏主结果文件的列结构）。"""
    if not area_stats:
        print("[提示] 未解析到面积分档结果（val.py 可能未开启 --area-bins），跳过分档 CSV。")
        return
    try:
        with open(AREA_RESULT_FILE, 'w', encoding='utf-8-sig') as f:
            f.write('run,weights,bin,nGT,P,R,mAP@.5,mAP@.5:.95,AR\n')
            for r in results:
                if r['metrics'] is None:
                    continue
                w = r['weights'].replace(',', ';')
                for b, v in r['metrics'].get('area', {}).items():
                    f.write(f"{r['run']},{w},{b},{v['nGT']},{v['P']:.4f},{v['R']:.4f},"
                            f"{v['mAP@.5']:.4f},{v['mAP@.5:.95']:.4f},{v['AR']:.4f}\n")
            for b, st in area_stats.items():
                m, sd = st['means'], st['stds']
                f.write(f"mean,,{b},{st['nGT']},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{m[3]:.4f},{m[4]:.4f}\n")
                f.write(f"std,,{b},{st['nGT']},{sd[0]:.4f},{sd[1]:.4f},{sd[2]:.4f},{sd[3]:.4f},{sd[4]:.4f}\n")
        print(f"分档结果已保存到: {AREA_RESULT_FILE}")
    except Exception as e:
        print(f"[警告] 保存分档结果文件失败: {e}")


def main():
    total_runs = len(WEIGHTS_LIST)
    print(f"准备评估 {total_runs} 个权重")
    print(f"项目目录: {ROOT}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    for i, weights in enumerate(WEIGHTS_LIST, 1):
        success, metrics, duration, _ = run_val_once(weights, i, total_runs)
        results.append({'run': i, 'weights': weights, 'metrics': metrics, 'duration': duration})
        # 非最后一次，等待几秒释放资源
        if i < total_runs:
            wait_time = 5
            print(f"等待 {wait_time} 秒后开始下一次评估...\n")
            time.sleep(wait_time)

    success_runs = [r for r in results if r['metrics'] is not None]
    failed_runs = [r for r in results if r['metrics'] is None]

    print(f"\n{'=' * 70}")
    print("评估汇总（整体 + 每个类别）")
    print(f"{'=' * 70}")

    all_stats = None
    class_stats = {}

    # ---- 整体表 ----
    all_rows = [{'run': r['run'], 'metrics': r['metrics']['all']} for r in success_runs]
    if all_rows:
        arr = np.array([[r['metrics'][k] for k in METRIC_NAMES] for r in all_rows], dtype=float)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0, ddof=1) if len(all_rows) > 1 else np.zeros(4)
        all_stats = {'means': means, 'stds': stds}
        print_metric_table('【整体 all】', all_rows, means, stds)

    # ---- 各类别表（类别名按输出顺序收集）----
    class_names = []
    for r in success_runs:
        for cls in r['metrics']['classes']:
            if cls not in class_names:
                class_names.append(cls)

    for cls in class_names:
        cls_rows = [{'run': r['run'], 'metrics': r['metrics']['classes'][cls]}
                    for r in success_runs if cls in r['metrics']['classes']]
        if not cls_rows:
            continue
        arr = np.array([[r['metrics'][k] for k in METRIC_NAMES] for r in cls_rows], dtype=float)
        means = arr.mean(axis=0)
        stds = arr.std(axis=0, ddof=1) if len(cls_rows) > 1 else np.zeros(4)
        class_stats[cls] = {'means': means, 'stds': stds}
        print_metric_table(f'【类别 {cls}】', cls_rows, means, stds)

    # ---- 面积分档 AP（AP_S / AP_M / AP_L / AR）----
    area_bins = []
    for r in success_runs:
        for b in r['metrics'].get('area', {}):
            if b not in area_bins:
                area_bins.append(b)

    area_stats = {}
    for b in area_bins:
        rows = [{'run': r['run'], 'metrics': r['metrics']['area'][b]}
                for r in success_runs if b in r['metrics'].get('area', {})]
        if not rows:
            continue
        arr = np.array([[r['metrics'][k] for k in AREA_METRICS] for r in rows], dtype=float)
        means = np.nanmean(arr, axis=0)
        stds = np.nanstd(arr, axis=0, ddof=1) if len(rows) > 1 else np.zeros(len(AREA_METRICS))
        area_stats[b] = {'means': means, 'stds': stds,
                         'nGT': int(rows[0]['metrics'].get('nGT', 0)), 'runs': len(rows)}

    for _prefix in ('COCO', 'scale'):
        _bins = [b for b in area_bins if b.startswith(_prefix + '-')]
        if _bins:
            print_area_table(f'【分档 AP · {_prefix}】', _bins, area_stats)

    # ---- Mean±Std 一行汇总 ----
    if all_stats is not None:
        print('\nMean±Std（整体）: ' + '  '.join(
            f"{m:.4f}±{s:.4f}" for m, s in zip(all_stats['means'], all_stats['stds'])))
        for cls, st in class_stats.items():
            print(f'Mean±Std（{cls}）: ' + '  '.join(
                f"{m:.4f}±{s:.4f}" for m, s in zip(st['means'], st['stds'])))

    if all_stats is not None:
        save_results(results, all_stats, class_stats)
        save_area_results(results, area_stats)
    else:
        print("所有评估均失败，请检查权重路径和参数。")

    print(f"\n成功评估: {len(success_runs)}/{total_runs}, 失败: {len(failed_runs)}")
    for r in failed_runs:
        print(f"  - 第 {r['run']} 次失败: {r['weights']}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()