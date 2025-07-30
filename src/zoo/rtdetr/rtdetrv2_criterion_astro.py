"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.distributed
import torch.nn.functional as F 
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ...core import register


@register()
class RTDETRCriterionv2(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
        matcher, 
        weight_dict, 
        losses, 
        alpha=0.2, 
        gamma=2.0, 
        num_classes=80, # Dsg의 클래스 수 (star/galaxy)
        boxes_weight_format=None,
        share_matched_indices=False,
        # --- Astro-YOLO: 새로운 인자 추가 ---
        galaxy_class_id=1,  # 'galaxy'에 해당하는 클래스 ID (논문에서 label 1)
        ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            boxes_weight_format: format for boxes weight (iou, )
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses 
        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        # --- Astro-YOLO: 인스턴스 변수 저장 ---
        self.galaxy_class_id = galaxy_class_id
        # ------------------------------------

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, # self.num_classes는 no-object 클래스
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        # num_classes+1: self.num_classes는 no-object, 실제 클래스는 0~num_classes-1
        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx]
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(\
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    # --- Astro-YOLO: 은하 타입 분류 손실 함수 추가 ---
    def loss_galaxy_types(self, outputs, targets, indices, num_boxes):
        """Compute the classification loss for galaxy types (smooth vs disk).
        Targets are filtered to only include 'galaxy' objects from the ground truth.
        """
        assert 'pred_galaxy_type_logits' in outputs
        src_logits = outputs['pred_galaxy_type_logits'] # (bs, num_queries, num_galaxy_types)

        idx = self._get_src_permutation_idx(indices) # (batch_idx, query_idx)

        # Ground Truth에서 매칭된 객체들의 원본 클래스 (star/galaxy)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Ground Truth에서 매칭된 객체들의 은하 타입 레이블 (smooth/disk)
        # 이 필드는 D_ds 데이터에만 유효합니다. D_sg 데이터에는 없거나 -1 등으로 채워져야 합니다.
        target_galaxy_types = torch.cat([t["galaxy_types"][J] for t, (_, J) in zip(targets, indices)])

        # '은하'로 분류된 (Ground Truth 기준) 객체만 필터링합니다.
        # 즉, target_classes_o가 self.galaxy_class_id와 일치하는 경우
        is_galaxy_gt = (target_classes_o == self.galaxy_class_id)
        
        # '은하'인 Ground Truth 객체에 해당하는 예측 로짓과 실제 타입 레이블을 가져옵니다.
        src_logits_galaxy = src_logits[idx][is_galaxy_gt]
        target_galaxy_types_galaxy = target_galaxy_types[is_galaxy_gt]

        if src_logits_galaxy.numel() == 0:
            # 매칭된 '은하' 객체가 없는 경우 손실은 0
            # require_grad=True를 추가하여 손실이 0이더라도 계산 그래프에 포함되도록 합니다.
            loss_galaxy_type = torch.tensor(0.0, device=src_logits.device, requires_grad=True)
        else:
            # Cross-Entropy 손실 계산
            loss_galaxy_type = F.cross_entropy(src_logits_galaxy, target_galaxy_types_galaxy,
                                              reduction='mean')

        return {'loss_galaxy_type': loss_galaxy_type}
    # ----------------------------------------------------

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            # --- Astro-YOLO: 새로운 손실 함수 추가 ---
            'galaxy_types': self.loss_galaxy_types,
            # ------------------------------------
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Retrieve the matching between the outputs of the last layer and the targets
        matched = self.matcher(outputs_without_aux, targets)
        indices = matched['indices']

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        #     meta = self.get_loss_meta_info(loss, outputs, targets, indices)            
        #     l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes, **meta)
        #     l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
        #     losses.update(l_dict)
        for loss_name in self.losses: # loss_name으로 변경하여 'loss' 변수와 충돌 방지
            meta = self.get_loss_meta_info(loss_name, outputs, targets, indices)            
            l_dict = self.get_loss(loss_name, outputs, targets, indices, num_boxes, **meta)
            # l_dict에서 k에 해당하는 가중치만 적용
            l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict} # .get(k, 1.0) 추가하여 weight_dict에 없는 경우 처리
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if not self.share_matched_indices:
                    matched = self.matcher(aux_outputs, targets)
                    indices = matched['indices']
                # for loss in self.losses:
                #     meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                #     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **meta)
                #     l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                #     l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                #     losses.update(l_dict)
                for loss_name in self.losses: # loss_name으로 변경
                    meta = self.get_loss_meta_info(loss_name, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss_name, aux_outputs, targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict} # .get(k, 1.0) 추가
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']):
                # for loss in self.losses:
                #     meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices)
                #     l_dict = self.get_loss(loss, aux_outputs, targets, indices, dn_num_boxes, **meta)
                #     l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                #     l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                #     losses.update(l_dict)
                for loss_name in self.losses: # loss_name으로 변경
                    meta = self.get_loss_meta_info(loss_name, aux_outputs, targets, indices)
                    l_dict = self.get_loss(loss_name, aux_outputs, targets, indices, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict} # .get(k, 1.0) 추가
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of encoder auxiliary losses. For rtdetr v2
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                # 인코더 보조 출력 매칭은 원본 targets 기준으로 수행 (클래스 재매핑 필요 없음)
                matched = self.matcher(aux_outputs, targets)    # enc_targets 대신 targets 사용
                indices = matched['indices']
                # for loss in self.losses:    
                #     meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices)
                #     l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices, num_boxes, **meta)
                #     l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                #     l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                #     losses.update(l_dict)
                for loss_name in self.losses: # loss_name으로 변경
                    # 인코더 보조 손실 계산 시, galaxy_types 손실은 포함되지 않도록 처리
                    # loss_name이 'galaxy_types'인 경우 건너뛰거나 가중치를 0으로 설정
                    if loss_name == 'galaxy_types':
                        continue # 인코더는 은하 타입 분류 헤드가 없으므로 건너뜀

                    meta = self.get_loss_meta_info(loss_name, aux_outputs, enc_targets, indices)
                    l_dict = self.get_loss(loss_name, aux_outputs, enc_targets, indices, num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict.get(k, 1.0) for k in l_dict} # .get(k, 1.0) 추가
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
            
            if class_agnostic:
                self.num_classes = orig_num_classes

        # return losses
        
        # 최종 반환 시, weight_dict에 있는 키만 합산하도록 변경
        # 이렇게 하면 det_engine.py에서 weight_dict를 동적으로 조절할 때 효과적입니다.
        total_loss = sum(losses[k] for k in losses.keys() if k in self.weight_dict and self.weight_dict[k] > 0)
        return total_loss # 단일 손실 값을 반환 (기존 코드의 `losses`와 일치)


    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou(\
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes', ):
            meta = {'boxes_weight': iou}
        elif loss in ('vfl', ):
            meta = {'values': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device
        
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                    torch.zeros(0, dtype=torch.int64,  device=device)))
        
        return dn_match_indices
