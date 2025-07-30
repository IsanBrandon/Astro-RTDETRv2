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
        # 이 fallback은 'data_type' 필드가 확실히 추가된다면 필요 없지만, 견고성을 위해 유지합니다.
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
        # RTDETRCriterionv2의 forward 메서드가 이 weight_dict를 사용하여
        # 각 개별 손실에 가중치를 적용할 것이라고 가정합니다.
        criterion.weight_dict = current_loss_weights
        
        # ----------------------------------------------------

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                # criterion은 이제 개별 손실 딕셔너리를 반환하도록 수정되었으므로
                # 그대로 loss_dict에 받습니다.
                loss_dict = criterion(outputs, targets, **metas)

            # --- Astro-YOLO: loss_dict에서 총 손실 계산 ---
            # 개별 손실 딕셔너리에서 현재 배치에 활성화된 손실들에 가중치를 곱하여 합산합니다.
            # criterion.weight_dict는 현재 배치에 맞게 이미 설정되어 있으므로,
            # losses 딕셔너리의 키를 순회하며 criterion.weight_dict에 해당하는 가중치를 곱합니다.
            loss = sum(loss_dict[k] * current_loss_weights.get(k, 1.0) for k in loss_dict.keys() if k in current_loss_weights and current_loss_weights[k] > 0)
            # -----------------------------------------------
            
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            # criterion은 이제 개별 손실 딕셔너리를 반환합니다.
            loss_dict = criterion(outputs, targets, **metas)
            
            # --- Astro-YOLO: loss_dict에서 총 손실 계산 ---
            loss : torch.Tensor = sum(loss_dict[k] * current_loss_weights.get(k, 1.0) for k in loss_dict.keys() if k in current_loss_weights and current_loss_weights[k] > 0)
            # -----------------------------------------------

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

        # --- Astro-YOLO: 손실 로깅을 위한 수정 ---
        # loss_dict는 이제 개별 손실 값들을 담고 있습니다.
        # reduce_dict를 통해 분산 환경에서 손실을 평균화합니다.
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        
        # 모든 손실 값들을 스케일링된 형태로 로깅
        # `item()`을 사용하여 텐서를 파이썬 스칼라로 변환
        loss_dict_reduced_scaled = {k: v.item() * current_loss_weights.get(k, 1.0) for k, v in loss_dict_reduced.items() if k in current_loss_weights and current_loss_weights[k] > 0}
        
        # 스케일링되지 않은 원본 손실 값들을 로깅
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v.item() for k, v in loss_dict_reduced.items()}
        
        # 총 손실 값은 이미 위에서 계산된 'loss' 텐서의 item() 값입니다.
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced) # 개별 손실 딕셔너리를 출력
            sys.exit(1)

        # metric_logger.update에 스케일링된 총 손실과 개별 손실들을 전달
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value, global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items(): # 개별 손실 로깅
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
    
    # --- Astro-YOLO: 평가 시 criterion의 weight_dict를 원본으로 복원 ---
    # 훈련 모드에서 변경되었을 수 있으므로, 평가 시에는 모든 손실이 계산되도록 복원합니다.
    # criterion의 original_loss_weights가 criterion.__init__에서 설정되어 있다고 가정합니다.
    original_weights_for_eval = criterion.weight_dict.copy() # 현재 훈련 상태의 가중치 복사
    # 모든 손실에 대해 1.0 (또는 설정된 기본값)의 가중치를 부여한 새 딕셔너리 생성
    # 여기서 `criterion.losses`는 criterion이 계산해야 할 모든 손실의 이름 리스트입니다.
    eval_weight_dict = {loss_name: original_weights_for_eval.get(loss_name, 1.0) for loss_name in criterion.losses}
    
    # criterion의 weight_dict를 평가 모드용으로 설정
    criterion.weight_dict = eval_weight_dict
    # ------------------------------------------------------------------

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        
        results = postprocessor(outputs, orig_target_sizes)

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
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # --- Astro-YOLO: 평가 후 criterion.weight_dict를 원본 훈련 가중치로 복원 ---
    # 다음 훈련 에폭을 위해 weight_dict를 원본 상태로 돌려놓습니다.
    criterion.weight_dict = original_weights_for_eval
    # --------------------------------------------------------------------------

    return stats, coco_evaluator