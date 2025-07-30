"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
# 수정된 det_engine_astro 버전을 import 합니다.
from .det_engine_astro import train_one_epoch, evaluate 


class DetSolver(BaseSolver):
    # __inject__ 인자에 dataloader_ds를 추가하여 Dds 데이터로더를 주입받을 수 있도록 합니다.
    __inject__ = ['model', 'ema_model', 'ema', 'optimizer', 'scheduler',
                  'criterion', 'scaler', 'matcher', 'postprocessor',
                  'dataloader', 'metric_logger', 'dataloader_ds'] 

    def __init__(self,
                 model, ema_model=None, ema=None, optimizer=None,
                 scheduler=None, criterion=None, scaler=None,
                 matcher=None, postprocessor=None, dataloader=None, # 이 dataloader는 Dsg 훈련용으로 사용됩니다.
                 metric_logger=None,
                 dataloader_ds=None, # Dds 훈련용 데이터로더를 위한 인자
                 val_dataloader_ds=None, # Dds 평가용 데이터로더를 위한 인자 (선택 사항, 필요시 추가)
                 ):
        super().__init__(
            model, ema_model, optimizer, scheduler, criterion, scaler,
            matcher, postprocessor, dataloader, metric_logger # dataloader는 train_dataloader (Dsg용)
        )
        # self.dataloader는 이제 Dsg용 훈련 데이터로더입니다.
        self.train_dataloader_sg = dataloader 
        self.train_dataloader_ds = dataloader_ds 
        self.val_dataloader_ds = val_dataloader_ds 

        # criterion의 원본 weight_dict를 저장합니다.
        # train_one_epoch에서 이 original_loss_weights를 사용하여 동적으로 weight_dict를 설정합니다.
        self.original_loss_weights = self.criterion.weight_dict.copy() 

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):
            # --- Astro-YOLO: Dsg 데이터셋 훈련 단계 ---
            if self.train_dataloader_sg is not None:
                self.train_dataloader_sg.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.train_dataloader_sg.sampler.set_epoch(epoch)
                
                print(f"--- Training with D_sg data (Epoch {epoch}) ---")
                train_stats_sg = train_one_epoch(
                    self.model, 
                    self.criterion, 
                    self.train_dataloader_sg, # Dsg용 데이터로더 전달
                    self.optimizer, 
                    self.device, 
                    epoch, 
                    max_norm=args.clip_max_norm, 
                    print_freq=args.print_freq, 
                    ema=self.ema, 
                    scaler=self.scaler, 
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                    # Astro-YOLO: 현재 훈련 중인 데이터 타입 명시
                    current_data_type='D_sg' 
                )
            else:
                train_stats_sg = {} # Dsg 데이터로더가 없는 경우 빈 딕셔너리

            # --- Astro-YOLO: Dds 데이터셋 훈련 단계 ---
            if self.train_dataloader_ds is not None:
                self.train_dataloader_ds.set_epoch(epoch)
                if dist_utils.is_dist_available_and_initialized():
                    self.train_dataloader_ds.sampler.set_epoch(epoch)

                print(f"--- Training with D_ds data (Epoch {epoch}) ---")
                train_stats_ds = train_one_epoch(
                    self.model, 
                    self.criterion, 
                    self.train_dataloader_ds, # Dds용 데이터로더 전달
                    self.optimizer, 
                    self.device, 
                    epoch, 
                    max_norm=args.clip_max_norm, 
                    print_freq=args.print_freq, 
                    ema=self.ema, 
                    scaler=self.scaler, 
                    lr_warmup_scheduler=self.lr_warmup_scheduler,
                    writer=self.writer,
                    # Astro-YOLO: 현재 훈련 중인 데이터 타입 명시
                    current_data_type='D_ds'
                )
            else:
                train_stats_ds = {} # Dds 데이터로더가 없는 경우 빈 딕셔너리
            # -------------------------------------------------------------------

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # --- Astro-YOLO: 평가 시 criterion의 weight_dict를 원본으로 복원 ---
            # evaluate 함수는 모든 손실을 계산해야 하므로, weight_dict를 원본으로 설정합니다.
            self.criterion.weight_dict = self.original_loss_weights.copy()
            # -------------------------------------------------------------------

            module = self.ema.module if self.ema else self.model
            # Astro-YOLO: Dsg 평가와 Dds 평가를 분리하여 수행할 수 있습니다.
            # 여기서는 Dsg 평가를 기본으로 하고, Dds 평가가 필요하면 추가합니다.
            # val_dataloader는 Dsg 평가용으로 가정합니다.
            test_stats_sg, coco_evaluator_sg = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, # Dsg 평가용 데이터로더
                self.evaluator, # Dsg 평가기
                self.device
            )
            # Dds 평가 (필요시 추가)
            test_stats_ds = {}
            coco_evaluator_ds = None
            if self.val_dataloader_ds is not None:
                # Dds 평가용 evaluator를 별도로 생성해야 합니다.
                # (evaluator는 CocoEvaluator이므로 Dsg와 Dds 카테고리에 따라 달라집니다)
                # 여기서는 간단히 test_stats만 가져오는 것으로 처리
                test_stats_ds, coco_evaluator_ds = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader_ds, # Dds 평가용 데이터로더
                    # Dds용 evaluator를 별도로 주입받거나 생성해야 함
                    None, # 또는 Dds 전용 평가기 인스턴스
                    self.device
                )

            # TODO 
            for k in test_stats_sg: # Dsg 평가 통계 로깅
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats_sg[k]):
                        self.writer.add_scalar(f'Test_sg/{k}_{i}'.format(k), v, epoch) 
            if test_stats_ds: # Dds 평가 통계 로깅
                 for k in test_stats_ds:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats_ds[k]):
                            self.writer.add_scalar(f'Test_ds/{k}_{i}'.format(k), v, epoch) 

            # best_stat 로직은 주로 mAP 등 하나의 주요 메트릭을 추적합니다.
            # Astro-YOLO는 Dsg mAP와 Dds accuracy를 동시에 가지므로,
            # 어떤 메트릭을 'best_stat'으로 추적할지 결정해야 합니다.
            # 여기서는 Dsg의 bbox mAP를 추적하는 것으로 가정합니다.
            main_eval_metric = 'coco_eval_bbox' # Dsg의 주요 평가 메트릭
            if main_eval_metric in test_stats_sg:
                if main_eval_metric in best_stat:
                    best_stat['epoch'] = epoch if test_stats_sg[main_eval_metric][0] > best_stat[main_eval_metric] else best_stat['epoch']
                    best_stat[main_eval_metric] = max(best_stat[main_eval_metric], test_stats_sg[main_eval_metric][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[main_eval_metric] = test_stats_sg[main_eval_metric][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_sg.pth') 

            print(f'best_stat (Dsg bbox): {best_stat}')

            # --- Astro-YOLO: 로그 통계(`log_stats`) 업데이트 ---
            log_stats = {
                **{f'train_sg_{k}': v for k, v in train_stats_sg.items()}, # Dsg 훈련 통계
                **{f'train_ds_{k}': v for k, v in train_stats_ds.items()}, # Dds 훈련 통계
                **{f'test_sg_{k}': v for k, v in test_stats_sg.items()}, # Dsg 평가 통계
                **{f'test_ds_{k}': v for k, v in test_stats_ds.items()}, # Dds 평가 통계
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            # ----------------------------------------------------------

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator_sg is not None:
                    (self.output_dir / 'eval_sg').mkdir(exist_ok=True) 
                    if "bbox" in coco_evaluator_sg.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator_sg.coco_eval["bbox"].eval,
                                    self.output_dir / "eval_sg" / name) 
                if coco_evaluator_ds is not None: # Dds 평가 결과 저장
                    (self.output_dir / 'eval_ds').mkdir(exist_ok=True)
                    # Dds 평가 결과 저장 로직 (예: 정확도)
                    # torch.save(coco_evaluator_ds.some_eval_metric, self.output_dir / "eval_ds" / name)
            # -------------------------------------------------------------------------------------

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        # Astro-YOLO: 평가 시 criterion의 weight_dict를 원본으로 복원
        self.criterion.weight_dict = self.original_loss_weights.copy()
        # -------------------------------------------------------------------

        module = self.ema.module if self.ema else self.model
        # Dsg 평가
        test_stats_sg, coco_evaluator_sg = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device) # self.val_dataloader를 사용해야 함
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator_sg.coco_eval["bbox"].eval, self.output_dir / "eval_sg.pth") 
        
        # Dds 평가 (필요시 추가)
        if self.val_dataloader_ds is not None:
            test_stats_ds, coco_evaluator_ds = evaluate(module, self.criterion, self.postprocessor,
                    self.val_dataloader_ds, None, self.device) # Dds용 evaluator가 필요
            if self.output_dir:
                # dist_utils.save_on_master(coco_evaluator_ds.some_eval_metric, self.output_dir / "eval_ds.pth")
                pass # Dds 평가 저장 로직

        return