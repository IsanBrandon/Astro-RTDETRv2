""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import cv2
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt  # avoid name clash


import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms.v2 import functional as F
from torchvision.utils import draw_bounding_boxes

import PIL

__all__ = ['show_sample', 'draw_detections_on_image', 'visualize_batch_predictions']


def _label_to_color(label: int):
    """
    Galaxy = 0 → Blue, Star = 1 → Orange (BGR in OpenCV)
    """
    if label == 0:   # galaxy
        return (255, 0, 0)       # Blue
    elif label == 1: # star
        return (0, 165, 255)     # Orange
    else:
        return (0, 255, 0)       # fallback: green


def draw_detections_on_image(img_bgr: np.ndarray,
                             boxes_xyxy: np.ndarray,
                             labels: np.ndarray,
                             scores: np.ndarray,
                             class_names=('galaxy', 'star'),
                             score_thr: float = 0.25):
    """
    img_bgr: HxWx3 uint8 (BGR)
    boxes_xyxy: Nx4 (float, pixel xyxy)
    labels: N (int)
    scores: N (float)
    """
    H, W = img_bgr.shape[:2]

    # 간혹 잘못된 박스가 있을 수 있으므로 필터 (x2>x1, y2>y1, 화면 안)
    if len(boxes_xyxy) > 0:
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        valid = (x2 > x1) & (y2 > y1) & (x2 > 0) & (y2 > 0) & (x1 < W) & (y1 < H)
        boxes_xyxy = boxes_xyxy[valid]
        labels = labels[valid]
        scores = scores[valid]

    for box, lab, sc in zip(boxes_xyxy, labels, scores):
        if sc < score_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color = _label_to_color(int(lab))

        thickness = max(1, min(3, (y2 - y1) // 40))  # 박스 크기에 비례한 테두리 두께
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
        cname = class_names[int(lab)] if 0 <= int(lab) < len(class_names) else str(int(lab))
        box_h = max(1, y2 - y1)  # ← 먼저 계산
        caption = cname if box_h < 20 else f"{cname}:{sc:.2f}"
        font_scale = max(0.3, min(0.6, box_h / 80.0))  # 높이 비례
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        
        y_text = y1 - 3
        # 위로 나가면 박스 아래로 이동
        if y_text - th - 4 < 0:
            y_text = min(H - 1, y2 + th + 6)
        cv2.rectangle(img_bgr, (x1, y_text - th - 4), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(img_bgr, caption, (x1 + 2, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1, cv2.LINE_AA)
    return img_bgr


def show_sample(sample):
    """for coco dataset/dataloader (sanity check)"""

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()


@torch.no_grad()
def visualize_batch_predictions(model,
                                postprocessor,
                                batch_images: torch.Tensor,
                                device: torch.device,
                                save_dir: str,
                                class_names=('galaxy', 'star'),
                                score_thr: float = 0.5,
                                max_images: int = 16):
    """
    - batch_images: tensor [B, C, H, W] (0~1 float32 권장; 아니라면 아래 스케일 변환 줄 변경)
    - postprocessor: outputs, orig_sizes -> list[{'boxes','labels','scores'}]
    - 저장: save_dir/vis_XXX.jpg
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    imgs = batch_images.to(device, non_blocking=True)
    B, _, H, W = imgs.shape

    # forward
    outputs = model(imgs)

    # postprocessor는 보통 '현재 입력 해상도(H,W)' 기준 픽셀값 출력을 원함
    orig_sizes = torch.as_tensor([[H, W]] * B, dtype=torch.long, device=device)
    results = postprocessor(outputs, orig_sizes)  # list of dict

    # 텐서 → uint8 이미지
    # (학습 파이프라인에서 ConvertPILImage(scale=True)이면 0~1 범위)
    imgs_uint8 = (imgs.clamp(0, 1).mul(255).byte().cpu().numpy())

    for i in range(min(B, max_images)):
        img_rgb = np.transpose(imgs_uint8[i], (1, 2, 0))  # CHW->HWC
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        boxes = results[i]["boxes"].detach().cpu().numpy()
        labels = results[i]["labels"].detach().cpu().numpy()
        scores = results[i]["scores"].detach().cpu().numpy()

        img_vis = draw_detections_on_image(img_bgr, boxes, labels, scores,
                                           class_names=class_names, score_thr=score_thr)
        cv2.imwrite(os.path.join(save_dir, f"vis_{i:03d}.jpg"), img_vis)
