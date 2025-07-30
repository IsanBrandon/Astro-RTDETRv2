"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)
    
    # --- Astro-YOLO: 원본 손실 가중치를 저장합니다. ---
    # criterion.weight_dict는 RTDETRCriterionv2.__init__에서 설정된 기본 가중치입니다.
    original_loss_weights = criterion.weight_dict.copy()
    # ----------------------------------------------------

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        # --- Astro-YOLO: 데이터셋 타입에 따른 손실 가중치 동적 조절 ---
        # targets[0]에 'data_type' 키가 있다고 가정합니다 (데이터 로더에서 추가).
        # 'galaxy_types' 키의 존재 여부로도 데이터셋 타입을 추론할 수 있습니다.
        is_ds_data = False
        if targets and len(targets) > 0 and 'data_type' in targets[0]:
            if targets[0]['data_type'] == 'D_ds':
                is_ds_data = True
            elif targets[0]['data_type'] == 'D_sg':
                is_ds_data = False
        # Fallback: targets에 'galaxy_types'가 있고 (D_sg 데이터에서는 없을 수 있음)
        # 해당 배치의 첫 번째 타겟에 galaxy_types가 유효한 데이터로 들어있다면 D_ds로 간주
        elif targets and len(targets) > 0 and 'galaxy_types' in targets[0] and len(targets[0]['galaxy_types']) > 0:
             is_ds_data = True


        current_loss_weights = original_loss_weights.copy() # 매 배치마다 초기화

        if is_ds_data:
            # D_ds 데이터인 경우:
            # - loss_focal (1차 분류: star/galaxy) 가중치를 0으로 설정
            # - loss_boxes, loss_giou (탐지) 가중치를 0으로 설정
            # - loss_galaxy_type (2차 분류: smooth/disk) 가중치를 유지
            current_loss_weights['loss_focal'] = 0.0 # 혹은 'loss_labels' 계열
            current_loss_weights['loss_bbox'] = 0.0
            current_loss_weights['loss_giou'] = 0.0
            # 보조 손실에도 동일하게 적용 (접미사 _aux_X, _dn_X, _enc_X 고려)
            for k in list(current_loss_weights.keys()):
                if '_aux_' in k or '_dn_' in k or '_enc_' in k:
                    if 'loss_focal' in k or 'loss_bbox' in k or 'loss_giou' in k:
                        current_loss_weights[k] = 0.0

        else:
            # D_sg 데이터인 경우:
            # - loss_galaxy_type (2차 분류) 가중치를 0으로 설정
            current_loss_weights['loss_galaxy_type'] = 0.0
            for k in list(current_loss_weights.keys()):
                if '_aux_' in k or '_dn_' in k or '_enc_' in k:
                    if 'loss_galaxy_type' in k:
                        current_loss_weights[k] = 0.0
        
        # criterion의 weight_dict를 동적으로 조절
        criterion.weight_dict = current_loss_weights
        
        # ----------------------------------------------------

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            
            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    return stats, coco_evaluator



