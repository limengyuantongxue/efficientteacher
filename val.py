# EfficientTeacher
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.backbone.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.metrics import ap_per_class, ConfusionMatrix, oks_iou
from utils.general import coco80_to_coco91_class, check_dataset, check_img_size, check_requirements, \
    check_suffix, check_yaml, box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, \
    increment_path, colorstr, print_args, non_max_suppression_lmk_and_bbox, scale_coords_landmarks
from utils.plots import plot_images
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.profile import profile
from utils.torch_utils import is_parallel
import copy
from configs.defaults import get_cfg

from utils.metrics import NMEMeter
from utils.detect_multi_backend import DetectMultiBackend

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})



def process_batch_oks(detections, labels, iouv, num_points):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    correct_class = labels[:, 0:1] == detections[:, 5]
    ious = oks_iou(labels, detections, num_points)
    ious = torch.from_numpy(ious).to(iouv.device)

    for i in range(len(iouv)):
        x = torch.where((ious >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct

def process_batch_old(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # print('process batch:', detections.shape)
    # iou = box_iou(labels[:, 1:], xywh2xyxy(poly2hbb(detections[:, -9:-1])))
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        model_post=None,
        eval_num = -1,
        cfg = None,
        val_ssod = False,
        area_bins = 'none',  # 面积分档 AP: 'none' | 'coco'(32^2/96^2) | 'scale'(0.001/0.002*img^2) | 'both'
        num_points = 0,
        val_kp = False,
        val_dp1000 = False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        names = {}
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
        pt, jit, engine = True, False, False
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # check_suffix(weights, '.pt')
        # model = attempt_load(weights, map_location=device, fuse=True)  # load FP32 model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine, is_magicmind = model.stride, model.pt, model.jit, model.engine, model.magicmind
        gs = max(int(model.stride), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        if cfg != '':
            val_cfg = get_cfg()
            val_cfg.merge_from_file(cfg)
            data = {}
            data['val'] = val_cfg.Dataset.val
            data['nc'] = val_cfg.Dataset.nc
            data['names'] = val_cfg.Dataset.names
            val_kp = val_cfg.Dataset.val_kp
        else:
            data = check_dataset(data)  # check
            val_cfg = None

        # Data
        if val_ssod:
            pass
        elif pt: #only pt we print FLOPs and PARAMS
            model_profile = copy.deepcopy(model)
            flops, params =  profile(model_profile.module if is_parallel(model_profile) else model_profile, (torch.ones((1, 3, imgsz, imgsz)).to(device),1), clever=True)
            print("Flops {} Params {}".format(flops, params))
        else:
            pass


    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True, cfg=val_cfg,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    try:
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    except:
        names = names
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    # ==================== 面积分档 AP（AP_S / AP_M / AP_L / AR）====================
    # 口径（COCO 式）：GT 与预测框【同时】按面积分档过滤后再做匹配。
    #   'coco' : 每档边界 32^2 / 96^2 像素^2（COCO small/medium/large）
    #   'scale': 每档边界 0.001*img^2 / 0.002*img^2（与 SaA 的 scale_threshold、SDFEN 的 b1/b2 同口径）
    # 用途：为"面向小目标"提供直接证据（小目标档的 AP 才是本文关心的量）。
    # 注意：'scale' 档必须用【归一化面积占比】(w*h / 原图面积)，与 SaA 的 scale_threshold=0.001*img^2、
    #       SDFEN 的 b1=0.001/b2=0.002 完全同口径；若误用 native 像素面积，会因原图分辨率(1280)与
    #       训练分辨率(640)不同而整体放大 (1280/640)^2=4 倍，导致分档错位。
    area_sets = {}
    _ab = str(area_bins).lower()
    if _ab in ('coco', 'both'):
        area_sets['COCO'] = ((0.0, 32 ** 2, 96 ** 2, float('inf')), 'px')
    if _ab in ('scale', 'both'):
        area_sets['scale'] = ((0.0, 0.001, 0.002, float('inf')), 'norm')
    area_stats = {k: [[], [], []] for k in area_sets}      # 每档: [(correct, conf, pcls, tcls), ...]
    area_gt_count = {k: [0, 0, 0] for k in area_sets}

    def _accumulate_area(pred, predn, labelsn, shape=None):
        """把当前图像的预测/标签按面积分档累计 TP/FP。

        pred / predn : 同一预测集合（predn 为 native-space 坐标）；pred 可为空张量
        labelsn      : native-space 标签 (n,5) = [cls, x1,y1,x2,y2]；None 表示该图无 GT
        说明：某档内若无 GT，则该档的预测全部记为该档 FP；无预测时只登记 GT（保证 recall 分母完整）。
        """
        _imgarea = float(shape[0]) * float(shape[1]) if shape is not None else None
        parea_px = None if len(pred) == 0 else \
            (predn[:, 2] - predn[:, 0]).clamp(min=0) * (predn[:, 3] - predn[:, 1]).clamp(min=0)
        tarea_px = None if (labelsn is None or len(labelsn) == 0) else \
            (labelsn[:, 3] - labelsn[:, 1]).clamp(min=0) * (labelsn[:, 4] - labelsn[:, 2]).clamp(min=0)
        # 归一化面积占比：native 像素面积 / 原图面积（等价于 SaA 用的 w_norm*h_norm）
        parea_nm = (parea_px / _imgarea) if (parea_px is not None and _imgarea) else None
        tarea_nm = (tarea_px / _imgarea) if (tarea_px is not None and _imgarea) else None
        for _key, (_edges, _mode) in area_sets.items():
            tarea = tarea_nm if _mode == 'norm' else tarea_px
            parea = parea_nm if _mode == 'norm' else parea_px
            for _bi in range(3):
                gk = ((tarea >= _edges[_bi]) & (tarea < _edges[_bi + 1])) if tarea is not None else None
                pk = ((parea >= _edges[_bi]) & (parea < _edges[_bi + 1])) if parea is not None else None
                n_g = int(gk.sum()) if gk is not None else 0
                n_p = int(pk.sum()) if pk is not None else 0
                if n_g == 0 and n_p == 0:
                    continue
                area_gt_count[_key][_bi] += n_g
                if n_g:
                    c = process_batch(predn[pk], labelsn[gk], iouv).cpu()   # pk 可为空 -> (0, niou)
                    tcl = labelsn[gk][:, 0].tolist()
                else:                       # 该档无 GT：这些预测全算 FP
                    c = torch.zeros(n_p, niou, dtype=torch.bool)
                    tcl = []
                area_stats[_key][_bi].append(
                    (c.numpy(), pred[pk, 4].cpu().numpy(), pred[pk, 5].cpu().numpy(), tcl))
    # =============================================================================

    ##model_postprocess
    # from models.head.yolov5_head import Detect
    # model_post = Detect(cfg)
    # model_post.training = False
    # model_post.eval()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 默认-1 则全部测试
        if batch_i==eval_num:
            break
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if pt:
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        elif is_magicmind: #for cambricon no need to divide 255.0 
            pass
        else:
            img /= 255.0   
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        if val_ssod:
            outputs, sup_feats = model(img, augment=augment) 
        else:
            outputs = model(img, augment=augment)  # inference and training outputs
        if model_post is not None:
            model_post.cur_imgsize = img.shape[2:]
            if is_magicmind:
                outputs = model_post.post_process_v2(outputs)
            else:
                outputs = model_post.post_process(outputs)
        dt[1] += time_sync() - t2

        #ugly solution suit for different output format
        if outputs is not list:
            out = outputs
            if type(outputs) is tuple:
                out = outputs[0]
            if len(outputs) == 2:
                out = outputs[0]
        else:
            if len(outputs) >= 2:
                out, train_out = outputs[:2]
            # Compute loss
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
            else:
                out = outputs[0]
        # Run NMS
        # if num_points == 4:
            # targets[:, 2:] *= torch.Tensor([width, height, width, height, width, height, width, height, width, height, width, height]).to(device)  # to pixels
        # if num_points == 8:
        if val_kp:
            targets[:, 2:2+2*(2 + num_points)] *= torch.Tensor([width, height] * (2 + num_points)).to(device)  # to pixels
        else:
            targets[:, 2:6] *= torch.Tensor([width, height] * 2).to(device)  # to pixels
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:6] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        if num_points > 0:
            out = non_max_suppression_lmk_and_bbox(out, conf_thres, iou_thres, labels=lb, agnostic=single_cls, num_points=num_points)
        else:
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3
        # print(time_sync() - t3)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    if area_sets and num_points == 0:   # 无预测：各档只登记 GT，保证 recall 分母完整
                        _tbox = xywh2xyxy(labels[:, 1:5])
                        scale_coords(img[si].shape[1:], _tbox, shape, shapes[si][1])
                        _accumulate_area(pred, pred, torch.cat((labels[:, 0:1], _tbox), 1), shape)
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            if num_points > 0:
                scale_coords_landmarks(img[si].shape[1:], predn[:, -1 - num_points * 2:-1], shape, num_points, shapes[si][1])

            # Evaluate
            if nl:
                if num_points > 0 and val_kp:
                    scale_coords_landmarks(img[si].shape[1:], labels[:, 5:5+num_points * 2], shape, num_points, shapes[si][1])
                    correct = process_batch_oks(predn, labels, iouv, num_points)
                else:
                    #虽然检测出了关键点，但是val_kp置0，所以只验bbox
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if area_sets:
                        _accumulate_area(pred, predn, labelsn, shape)

                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if area_sets and len(pred):     # 无 GT 图：这些预测在各档内均算 FP
                    _accumulate_area(pred, predn, None, shape)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            if val_kp:
                Thread(target=plot_images, args=(img, targets, paths, f, num_points, names), daemon=True).start()
            else:
                Thread(target=plot_images, args=(img, targets, paths, f, 0, names), daemon=True).start()
            # Thread(target=plot_images_keypoints, args=(img, targets, paths, f, names), daemon=True).start()
            # f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            # Thread(target=plot_images, args=(img, output_to_target_keypoints(out), paths, f, names), daemon=True).start()
            # Thread(target=plot_images, args=(img, out, paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, cls_thr = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        cls_thr = []
        nt = torch.zeros(1)
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print area-bin AP（AP_S / AP_M / AP_L）：面向小目标的直接证据
    # 注意：为避免被 scripts/run_val_multiple_times.py 的逐行解析误判为"类别行"，
    #       每档只打印 6 个字段（< 7 时该脚本会跳过该行）。
    if area_sets:
        print('\n' + colorstr('bold', 'Area-AP:') +
                    ' 面积分档 AP（GT 与预测框同时按面积过滤，COCO 口径）  imgsz=%d' % imgsz)
        print('  %-10s %8s %9s %9s %9s %9s %9s' %
                    ('bin', '#GT', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'AR@.5:.95'))
        for _key, (_edges, _mode) in area_sets.items():
            for _bi, _bn in enumerate('SML'):
                _ent = area_stats[_key][_bi]
                if not _ent:
                    continue
                _sarr = [np.concatenate(x, 0) for x in zip(*_ent)]
                if not (len(_sarr) and _sarr[0].any()):
                    # 该档无任何 TP：全部按 0 输出（保持与本表其它行一致的数值格式，便于脚本解析）
                    print('  %-10s %8d %9.4f %9.4f %9.4f %9.4f %9.4f' %
                                ('%s-%s' % (_key, _bn), area_gt_count[_key][_bi], 0.0, 0.0, 0.0, 0.0, 0.0))
                    continue
                _pb, _rb, _apb, _f1b, _apcb, _ = ap_per_class(*_sarr, plot=False, save_dir=save_dir, names=names)
                _ap50b, _apb_all = _apb[:, 0], _apb.mean(1)
                # AR = 各 IoU 阈值下"不加置信度过滤"的最大召回率的均值（召回上界）
                _ar = float(_sarr[0].sum(0).mean() / max(len(_sarr[3]), 1))
                print('  %-10s %8d %9.4f %9.4f %9.4f %9.4f %9.4f' %
                            ('%s-%s' % (_key, _bn), area_gt_count[_key][_bi],
                             _pb.mean(), _rb.mean(), _ap50b.mean(), _apb_all.mean(), _ar))
            if _mode == 'norm':
                print('  %s 档边界(归一化面积占比 w*h): %s' %
                      (_key, ' < '.join(['%.4f' % _edges[0]] +
                                        ['%.4f' % e for e in _edges[1:-1]] + ['inf'])))
            else:
                print('  %s 档边界(native px^2): %s' %
                      (_key, ' < '.join(['%.0f' % _edges[0]] +
                                        ['%.0f' % e for e in _edges[1:-1]] + ['inf'])))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        anno_json = str('/mnt/bowen/exp/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    if val_ssod:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, cls_thr
    else:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT/'pencil_coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="D:/个人文件/文献/毕业论文参考/code/efficientteacher/exp_ssod/exp58/weights/best.pt", help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=48, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--val-ssod',  action='store_true', help='trigger when val semi-supervised') 
    parser.add_argument('--area-bins', type=str, default='both',
                        help="area-bin AP: 'none' | 'coco'(32^2/96^2 px^2) | 'scale'(0.001/0.002*img^2) | 'both'")
    parser.add_argument('--num-points',  type=int, default=0, help='num of keypoints') 
    parser.add_argument('--cfg', type=str, default=r'D:\个人文件\文献\毕业论文参考\code\efficientteacher\configs\ssod\custom\yolov5m_custom_ssod_sdfen.yaml', help='The config file used for validation') 
    parser.add_argument('--val-dp1000',  action='store_true', help='trigger when val dp1000 model') 
    opt = parser.parse_args()
    if opt.cfg == '':
        opt.data = check_yaml(opt.data)  # check YAML
        opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    set_logging()
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
