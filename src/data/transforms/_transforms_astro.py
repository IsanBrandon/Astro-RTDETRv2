""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional, Tuple

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes

from ...core import register
# 박스 변환을 위한 유틸리티 import (RT-DETR의 util/box_ops.py에서 가져옴)
# 이제 box_ops.py 파일이 제공되었으므로, 해당 경로에서 import 합니다.
from ...zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh # box_ops import 
import random
import numpy as np


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
            
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)
    
    
# --- Astro-YOLO: Mosaic 증강 추가 ---
@register()
class Mosaic(T.Transform): # T.Transform을 상속받지만, 여러 이미지를 입력으로 받도록 __call__을 정의
    """
    Implements Mosaic data augmentation from YOLOv4.
    Combines 4 images into one.
    This transform expects a list of (image, target) tuples as input.
    """
    def __init__(self, output_size=(640, 640), p=0.5):
        super().__init__()
        self.output_size = output_size
        self.p = p

    def forward(self, img_target_list: List[Tuple[PIL.Image.Image, Dict[str, Any]]]):
        if torch.rand(1) >= self.p:
            # 확률에 따라 적용 안 할 경우, 첫 번째 이미지를 그대로 반환합니다.
            # 하지만 Mosaic은 4개의 이미지를 '필요'로 하므로, collate_fn에서
            # Mosaic을 적용할지 여부를 결정하고, 적용할 경우 4개의 이미지를 모아서 전달해야 합니다.
            # 여기서는 Transform의 인터페이스를 맞추기 위해 이렇게 처리합니다.
            if len(img_target_list) == 1:
                return img_target_list[0]
            else:
                # 콜레이트 함수에서 여러 이미지를 미리 모아 전달하는 경우, 첫 번째 것만 반환하는 것은 논리적 오류.
                # 따라서, Mosaic/Mixup은 Transforms.Compose 안에 넣는 것보다
                # 콜레이트 함수 내부에서 직접 호출하는 것이 더 적절합니다.
                # Transform 클래스로서의 정의는 유지하되, 호출 방식에 유의해야 합니다.
                return img_target_list[0] # 임시로 첫 번째 이미지 반환

        assert len(img_target_list) == 4, \
            "Mosaic augmentation requires exactly 4 (image, target) pairs."

        output_img = PIL.Image.new(
            img_target_list[0][0].mode, self.output_size, (128, 128, 128) # 회색 배경
        )
        
        # 임의의 중심점 선택 (mosaic 이미지의 4등분 경계점)
        # cx = random.randint(self.output_size[0] // 4, self.output_size[0] * 3 // 4)
        # cy = random.randint(self.output_size[1] // 4, self.output_size[1] * 3 // 4)
        # 간단화를 위해 중앙 고정
        cx = self.output_size[0] // 2
        cy = self.output_size[1] // 2


        new_targets_list = []
        for i, (img, targets) in enumerate(img_target_list):
            # 각 이미지의 위치 결정 (top-left, top-right, bottom-left, bottom-right)
            if i == 0: # Top-left
                x_offset, y_offset = 0, 0
                paste_coords = (0, 0)
                paste_size = (cx, cy)
            elif i == 1: # Top-right
                x_offset, y_offset = cx, 0
                paste_coords = (cx, 0)
                paste_size = (self.output_size[0] - cx, cy)
            elif i == 2: # Bottom-left
                x_offset, y_offset = 0, cy
                paste_coords = (0, cy)
                paste_size = (cx, self.output_size[1] - cy)
            elif i == 3: # Bottom-right
                x_offset, y_offset = cx, cy
                paste_coords = (cx, cy)
                paste_size = (self.output_size[0] - cx, self.output_size[1] - cy)
            
            # 이미지를 리사이즈하여 해당 영역에 붙여넣기
            # PIL 이미지를 (W, H)로 리사이즈
            img_resized = img.resize(paste_size, PIL.Image.BILINEAR)
            output_img.paste(img_resized, paste_coords)

            # 타겟 박스 좌표 변환 (cxcywh -> xyxy -> 픽셀 단위 -> 새로운 이미지 내에서 정규화된 cxcywh)
            if targets is not None and 'boxes' in targets and len(targets['boxes']) > 0:
                w_orig, h_orig = targets['orig_size'].tolist()

                boxes_cxcywh_norm = targets['boxes'] # 이미 0~1 정규화된 cxcywh라고 가정
                
                # --- Astro-YOLO: box_ops에서 import한 함수 사용 ---
                boxes_xyxy_norm = box_cxcywh_to_xyxy(boxes_cxcywh_norm) # 수정된 부분 
                # ----------------------------------------------------

                # 원본 픽셀 좌표로 변환
                boxes_xyxy_pixel = boxes_xyxy_norm * torch.tensor([w_orig, h_orig, w_orig, h_orig], dtype=torch.float32, device=boxes_xyxy_norm.device)

                # 모자이크 이미지 내에서의 픽셀 좌표로 오프셋 적용
                boxes_xyxy_pixel[:, 0] += x_offset
                boxes_xyxy_pixel[:, 1] += y_offset
                boxes_xyxy_pixel[:, 2] += x_offset
                boxes_xyxy_pixel[:, 3] += y_offset

                # 새로운 모자이크 이미지 크기에 맞춰 0~1 정규화
                boxes_xyxy_new_norm = boxes_xyxy_pixel / torch.tensor([self.output_size[0], self.output_size[1], self.output_size[0], self.output_size[1]], dtype=torch.float32, device=boxes_xyxy_pixel.device)

                # 클리핑 (새로운 이미지 영역을 벗어나는 박스 처리)
                boxes_xyxy_new_norm[:, [0, 2]] = boxes_xyxy_new_norm[:, [0, 2]].clamp(0, 1)
                boxes_xyxy_new_norm[:, [1, 3]] = boxes_xyxy_new_norm[:, [1, 3]].clamp(0, 1)

                # 유효한 박스만 유지 (너비/높이가 너무 작아지거나 0이 되는 경우 제거)
                keep = (boxes_xyxy_new_norm[:, 2] - boxes_xyxy_new_norm[:, 0] > 1e-3) & \
                       (boxes_xyxy_new_norm[:, 3] - boxes_xyxy_new_norm[:, 1] > 1e-3) # 아주 작은 박스 필터링

                boxes_xyxy_new_norm = boxes_xyxy_new_norm[keep]
                
                if boxes_xyxy_new_norm.numel() == 0:
                    continue # 유효한 박스가 없으면 다음 이미지로

                # 다시 cxcywh로 변환 (RT-DETR의 기본 형식)
                # --- Astro-YOLO: box_ops에서 import한 함수 사용 ---
                boxes_cxcywh_new = box_xyxy_to_cxcywh(boxes_xyxy_new_norm) # 수정된 부분 
                # ----------------------------------------------------
                
                # 새로운 targets 딕셔너리 생성 (labels, galaxy_types 포함)
                new_t = {}
                new_t['boxes'] = boxes_cxcywh_new
                new_t['labels'] = targets['labels'][keep]
                new_t['image_id'] = targets['image_id'] # image_id는 보통 그대로 유지
                new_t['orig_size'] = torch.as_tensor(self.output_size[::-1], dtype=torch.int64, device=targets['orig_size'].device) # 새로운 이미지 크기

                if 'galaxy_types' in targets: # galaxy_types 필드가 있는 경우에만 처리
                    new_t['galaxy_types'] = targets['galaxy_types'][keep]
                
                if 'data_type' in targets: # data_type 필드가 있는 경우에만 처리
                    new_t['data_type'] = targets['data_type']

                new_targets_list.append(new_t)

        # 모든 타겟 딕셔너리를 하나의 딕셔너리로 병합
        final_targets = {}
        if len(new_targets_list) > 0:
            for k in new_targets_list[0].keys():
                # Tensors만 concat하고, 스칼라나 문자열은 첫 번째 값 사용 (data_type, image_id 등)
                if isinstance(new_targets_list[0][k], torch.Tensor) and new_targets_list[0][k].dim() > 0:
                    final_targets[k] = torch.cat([t[k] for t in new_targets_list if k in t], dim=0)
                else:
                    final_targets[k] = new_targets_list[0][k] # 첫 번째 이미지의 값 사용 (예: image_id, data_type)
        else: # 모든 이미지가 유효한 박스를 생성하지 못한 경우
            # 이 경우, 빈 타겟 딕셔너리를 반환해야 합니다.
            # Criterion에서 이 빈 타겟을 잘 처리하도록 설계되어야 합니다.
            final_targets['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            final_targets['labels'] = torch.empty((0,), dtype=torch.int64)
            final_targets['image_id'] = img_target_list[0][1]['image_id'] # 기본 image_id는 유지
            final_targets['orig_size'] = torch.as_tensor(self.output_size[::-1], dtype=torch.int64)
            final_targets['data_type'] = img_target_list[0][1]['data_type'] # 기본 data_type은 유지
            # galaxy_types도 빈 텐서로 추가 (필요한 경우)
            if 'galaxy_types' in img_target_list[0][1]:
                final_targets['galaxy_types'] = torch.empty((0,), dtype=torch.int64)


        # PIL.Image를 RT-DETR의 Image 텐서 형식으로 변환 (ConvertPILImage와 유사)
        final_img = F.pil_to_tensor(output_img)
        final_img = final_img.float() / 255.
        final_img = Image(final_img) # RT-DETR의 Image 클래스 사용

        return final_img, final_targets


# --- Mixup 증강 추가 ---
@register()
class Mixup(T.Transform):
    """
    Implements Mixup data augmentation.
    Mixes two images and their labels/targets.
    This transform expects a list of 2 (image, target) tuples as input.
    """
    def __init__(self, alpha=1.5, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, img_target_list: List[Tuple[PIL.Image.Image, Dict[str, Any]]]):
        if torch.rand(1) >= self.p:
            if len(img_target_list) == 1:
                return img_target_list[0]
            else:
                return img_target_list[0] # 임시로 첫 번째 이미지 반환

        assert len(img_target_list) == 2, \
            "Mixup augmentation requires exactly 2 (image, target) pairs."

        img1, targets1 = img_target_list[0]
        img2, targets2 = img_target_list[1]

        # Convert PIL to Tensor for mixing
        # RT-DETR의 ConvertPILImage처럼 0-1 스케일링된 float32 텐서로 변환
        img1_tensor = F.pil_to_tensor(img1).float() / 255.
        img2_tensor = F.pil_to_tensor(img2).float() / 255.

        # Generate lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(0, min(1, lam)) # 0과 1 사이로 클리핑

        mixed_img_tensor = lam * img1_tensor + (1 - lam) * img2_tensor
        final_img = Image(mixed_img_tensor) # RT-DETR의 Image 클래스 사용

        # Concatenate targets (labels, boxes, galaxy_types)
        mixed_targets = {}
        for key in targets1.keys():
            if key in targets2 and isinstance(targets1[key], torch.Tensor) and targets1[key].dim() > 0:
                # 텐서 형태의 타겟 (boxes, labels, galaxy_types 등)
                mixed_targets[key] = torch.cat((targets1[key], targets2[key]), dim=0)
            else: 
                # 스칼라나 문자열 형태의 타겟 (image_id, orig_size, data_type)
                # Mixup은 일반적으로 동일 데이터셋 내에서 적용되므로, data_type은 동일하다고 가정.
                # orig_size는 두 이미지 중 하나로 통일.
                mixed_targets[key] = targets1[key] 
        
        return final_img, mixed_targets