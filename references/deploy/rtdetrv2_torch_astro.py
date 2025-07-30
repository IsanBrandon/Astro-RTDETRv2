"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import glob

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

from src.core import YAMLConfig


# draw 함수 정의
def draw(images, labels, boxes, scores, class_names, task, thrh = 0.6, output_filename='results.jpg'):
    for i, im in enumerate(images): # images는 항상 단일 이미지 리스트
        draw_obj = ImageDraw.Draw(im)

        # 1. Tensors to NumPy arrays and detach from graph
        scr = scores.detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()
        box = boxes.detach().cpu().numpy()

        # 2. Handle batch dimension (squeeze 0-th dimension if present)
        if scr.ndim == 2 and scr.shape[0] == 1:
            scr = scr.squeeze(0) # [1, N_detections] -> [N_detections]
        if lab.ndim == 2 and lab.shape[0] == 1:
            lab = lab.squeeze(0) # [1, N_detections] -> [N_detections]
        if box.ndim == 3 and box.shape[0] == 1:
            box = box.squeeze(0) # [1, N_detections, 4] -> [N_detections, 4]

        # 3. Filter detections by confidence threshold
        valid_indices = np.where(scr > thrh)[0]
        
        # 유효한 탐지가 없을 경우
        if len(valid_indices) == 0:
            print(f"No detections above threshold {thrh} for {os.path.basename(output_filename)}. Saving original image.")
            im.save(output_filename)
            continue # 다음 이미지로 넘어감

        valid_labels = lab[valid_indices]
        valid_boxes = box[valid_indices]

        for j in range(len(valid_labels)):
            # 4. Extract scalar label (ensure it's an int)
            label = int(valid_labels[j])

            # 5. Get box coordinates in [x1, y1, x2, y2] format
            b = valid_boxes[j].tolist()

            # PIL.ImageDraw.rectangle은 [x1, y1, x2, y2] 형식의 리스트를 기대하며,
            # 현재 b가 정확히 이 형식이므로 추가 변환 필요 없음.
            # 하지만, 혹시 모를 타입 오류에 대비하여 한번 더 확인하는 로직은 유지합니다.
            if not (len(b) == 4 and all(isinstance(coord, (int, float)) for coord in b)):
                # 이 경고 메시지는 유용할 수 있어 남겨두었습니다. 필요 없으면 제거하세요.
                print(f"Warning: Malformed box coordinates for detection {j} in {os.path.basename(output_filename)}. Expected [x1, y1, x2, y2] with 4 numbers, got {b}. Skipping this box.")
                continue # 잘못된 형식의 박스는 건너뜀

            outline_color = 'red' # 기본 색상

            # 6. Apply color based on task and class_id
            if class_names is not None and label in class_names:
                if task == 'dds': # Smooth vs. Disk (Dds)
                    # 클래스 이름 매핑: smooth (빨강), disk (초록)
                    if class_names.get(label) == 'smooth':
                        outline_color = 'red'
                    elif class_names.get(label) == 'disk':
                        outline_color = 'green'
                elif task == 'dsg': # Star vs. Galaxy (Dsg)
                    # 클래스 이름 매핑: star (주황), galaxy (파랑)
                    if class_names.get(label) == 'star':
                        outline_color = 'orange'
                    elif class_names.get(label) == 'galaxy':
                        outline_color = 'blue'
            
            # 7. Draw rectangle
            draw_obj.rectangle(b, outline=outline_color, width=2)

        # 8. Save image
        im.save(output_filename)


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    if args.im_folder:
        image_paths = glob.glob(os.path.join(args.im_folder, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(args.im_folder, '*.[pP][nN][gG]'))
        image_paths.sort()
    elif args.im_file:
        image_paths = [args.im_file]
    else:
        raise ValueError("Either --im-file or --im-folder must be specified.")

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.save_dir, exist_ok=True)

    # --- 클래스 이름 매핑 및 태스크 정의 (main 함수에서 처리) ---
    # AstroYOLO 논문 기반의 가정된 매핑
    
    # Dsg (Star vs. Galaxy) 태스크의 클래스 이름 및 매핑
    class_names_dsg = {0: 'star', 1: 'galaxy'} 
    
    # Dds (Smooth vs. Disk) 태스크의 클래스 이름 및 매핑
    class_names_dds = {0: 'smooth', 1: 'disk'} 

    task_to_visualize = None
    current_class_names = None

    # config 파일 이름에 따라 태스크 결정 로직 (basename 사용)
    config_base_name = os.path.basename(args.config)
    if 'rtdetrv2_dsg_star_galaxy' in config_base_name:
        task_to_visualize = 'dsg'
        current_class_names = class_names_dsg
        print(f"Detected task from config name: {task_to_visualize}. Using class names: {current_class_names}")
    elif 'rtdetrv2_dds' in config_base_name: 
        task_to_visualize = 'dds'
        current_class_names = class_names_dds
        print(f"Detected task from config name: {task_to_visualize}. Using class names: {current_class_names}")
    else:
        print(f"Warning: Could not determine task type from config path '{config_base_name}'. Please ensure your config filename contains 'rtdetrv2_dsg_star_galaxy' or 'rtdetrv2_dds'. Defaulting to DSG task and its class names.")
        task_to_visualize = 'dsg' 
        current_class_names = class_names_dsg 

    if current_class_names is None:
        raise ValueError("Class names could not be determined. Please check your config path or define class names explicitly.")
    # -------------------------------------------------------------------

    for im_idx, im_path in enumerate(image_paths):
        im_pil = Image.open(im_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        output_filename = os.path.join(args.save_dir, f'detected_{os.path.basename(im_path)}')
        
        draw([im_pil], labels, boxes, scores, 
             class_names=current_class_names, 
             task=task_to_visualize,          
             thrh=args.threshold, 
             output_filename=output_filename)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to the model checkpoint for inference.')
    parser.add_argument('--im-folder', type=str, help='Path to the folder containing images for inference.')
    parser.add_argument('-f', '--im-file', type=str, help='Path to a single image file for inference.')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to run inference on (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--save_dir', type=str, default='inference_results', help='Directory to save inference results.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for drawing bounding boxes.')
    args = parser.parse_args()
    main(args)