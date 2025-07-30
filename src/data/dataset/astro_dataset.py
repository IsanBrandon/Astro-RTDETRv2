# /home/uvll/Desktop/hyuk/rtdetr/rtdetrv2_pytorch/src/data/dataset/astro_dataset.py

import torch
import torchvision
from PIL import Image
from faster_coco_eval.utils.pytorch import FasterCocoDetection
from faster_coco_eval.core import mask as coco_mask
from ._dataset import DetDataset
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['AstroSgDataset', 'AstroDsDataset']

torchvision.disable_beta_transforms_warning()

# (이 부분은 coco_dataset.py에서 가져옴)
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]
            
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target

# COCO MSCOCO category mappings (필요시 사용)
# mscoco_category2name, mscoco_category2label, mscoco_label2category 도 필요시 여기에 추가


@register()
class AstroSgDataset(FasterCocoDetection, DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks

        # Dsg 데이터셋의 category_id 매핑 (star:0, galaxy:1)
        # FasterCocoDetection이 제공하는 categories 정보를 활용
        self.sg_category_to_label = {cat['id']: i for i, cat in enumerate(self.coco.dataset['categories'])}
        # 'star'와 'galaxy'의 실제 ID와 매핑된 레이블 확인 필요.
        # 일반적으로 COCO는 1부터 시작하는 ID를 사용하지만, Dsg는 0부터 시작하므로 확인 필수.
        # Dsg의 instances_train2017.json에서 categories가 [{id:0, name:star}, {id:1, name:galaxy}] 이므로
        # category_id 0 -> label 0 (star)
        # category_id 1 -> label 1 (galaxy)
        # 이는 ConvertCocoPolysToMask에서 category2label을 전달하지 않을 때 기본적으로 id를 그대로 사용합니다.
        # 따라서, self.sg_category_to_label은 {0:0, 1:1}이 될 것입니다.

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        
        # Dsg 데이터셋의 고유한 처리
        # target['labels']는 이미 star(0)/galaxy(1)로 되어 있음
        # D_sg 데이터에는 galaxy_types가 없으므로, -1 또는 None으로 설정
        target['galaxy_types'] = torch.full_like(target['labels'], -1, dtype=torch.int64) 
        
        # 데이터 타입 플래그 추가
        target['data_type'] = 'D_sg'

        if self._transforms is not None:
            # transforms는 image, target, self를 받으므로, image와 target만 넘겨주는 형태가 아님.
            # RT-DETR의 transforms는 `__call__(self, *inputs)` 형태이므로,
            # `img, target = self._transforms(img, target)` 로 호출해야 함.
            # CocoDetection의 __getitem__은 `img, target, _ = self._transforms(img, target, self)` 이므로
            # 여기서는 transform 후의 `_` (dataset) 반환값은 무시.
            img, target, _ = self._transforms(img, target, self) # 기존 코드 유지
        
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # Dsg는 remap_mscoco_category를 사용하지 않고 직접 label을 category_id에서 가져오도록
        # Prepare 함수에 self.sg_category_to_label을 전달하여 COCO id를 0/1로 매핑
        image, target = self.prepare(image, target, category2label=self.sg_category_to_label)
        
        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target: # Dsg는 마스크가 없을 가능성 높음
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        return image, target


@register()
class AstroDsDataset(FasterCocoDetection, DetDataset):
    __inject__ = ['transforms', ]

    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks

        # Dds 데이터셋의 category_id 매핑 (disk:0, smooth:1)
        self.ds_category_to_label = {cat['id']: i for i, cat in enumerate(self.coco.dataset['categories'])}
        # Dds의 instances_train2017.json에서 categories가 [{id:0, name:disk}, {id:1, name:smooth}] 이므로
        # category_id 0 -> label 0 (disk)
        # category_id 1 -> label 1 (smooth)
        # 이는 ConvertCocoPolysToMask에서 category2label을 전달하지 않을 때 기본적으로 id를 그대로 사용합니다.
        # 따라서, self.ds_category_to_label은 {0:0, 1:1}이 될 것입니다.

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        
        # Dds 데이터셋의 고유한 처리
        # target['labels']는 disk(0)/smooth(1)로 되어 있음
        # 여기서 D_ds 데이터는 'galaxy' 객체 자체를 탐지하는 것이 아니라,
        # 이미 '은하'로 판단된 객체의 '타입'을 분류하는 것이 목적이므로,
        # 'labels'는 모두 'galaxy'에 해당하는 ID (Dsg 기준 1)로 강제 변환합니다.
        # Astro-YOLO 논문의 목적: (1) 별/은하 탐지 및 분류, (2) 탐지된 은하를 smooth/disk 분류
        # Dds 데이터는 2차 분류에 사용되므로, 1차 분류 관점에서는 모두 'galaxy'여야 합니다.
        target['labels'] = torch.full_like(target['labels'], 1, dtype=torch.int64) # 'galaxy'의 ID = 1

        # target['galaxy_types']는 smooth(0)/disk(1) 값을 가짐
        # 원래 target['labels']에 있던 smooth/disk 정보를 galaxy_types로 옮깁니다.
        target['galaxy_types'] = target['labels'] # 이제 labels는 모두 galaxy, galaxy_types는 disk/smooth

        # 데이터 타입 플래그 추가
        target['data_type'] = 'D_ds'

        if self._transforms is not None:
            # Dds의 경우 transforms에 '중앙 크롭' 등 특화된 전처리가 포함되어야 할 수 있습니다.
            img, target, _ = self._transforms(img, target, self)
        
        return img, target

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # Prepare 함수에 self.ds_category_to_label을 전달하여 COCO id를 0/1로 매핑
        image, target = self.prepare(image, target, category2label=self.ds_category_to_label)
        
        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')
        
        return image, target