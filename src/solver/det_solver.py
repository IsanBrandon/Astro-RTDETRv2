"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()  # 이 메서드에서 _solver.py의 train 메서드가 호출되지만,
                      # 이제 해당 메서드에는 데이터로더 초기화 코드가 없습니다.
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')
        
        # 이제 fit 메서드에서 두 데이터로더를 명시적으로 초기화합니다.
        # YAML 설정에서 직접 shuffle 값을 가져와야 합니다.
        self.train_dataloader_dsg = dist_utils.warp_loader(self.cfg.train_dataloader, 
                                                            shuffle=self.cfg.yaml_cfg['train_dataloader'].get('shuffle', True))

        self.train_dataloader_dds = dist_utils.warp_loader(self.cfg.train_dataloader_dds,
                                                            shuffle=self.cfg.yaml_cfg['train_dataloader_dds'].get('shuffle', True))

        self.val_dataloader = dist_utils.warp_loader(self.cfg.val_dataloader,
                                                      shuffle=self.cfg.yaml_cfg['val_dataloader'].get('shuffle', False))
        
        # 👇 2. Mosaic 변환에 dataset 객체를 전달하는 로직을 추가합니다. 👇
        # Dsg 데이터셋
        transforms_dsg = self.train_dataloader_dsg.dataset.transforms
        if hasattr(transforms_dsg, 'ops') and transforms_dsg.ops[0].__class__.__name__ == 'Mosaic':
            transforms_dsg.ops[0].dataset = self.train_dataloader_dsg.dataset

        # Dds 데이터셋
        transforms_dds = self.train_dataloader_dds.dataset.transforms
        if hasattr(transforms_dds, 'ops') and transforms_dds.ops[0].__class__.__name__ == 'Mosaic':
            transforms_dds.ops[0].dataset = self.train_dataloader_dds.dataset

        best_stat = {'epoch': -1, }
        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):
            # 교대 훈련 로직
            if epoch % 2 == 0:
                # 짝수 에포크: Dsg 데이터셋으로 훈련 (star vs galaxy)
                current_dataloader = self.train_dataloader_dsg
                is_dsg_epoch = True
            else:
                # 홀수 에포크: Dds 데이터셋으로 훈련 (smooth vs disk)
                current_dataloader = self.train_dataloader_dds
                is_dsg_epoch = False

            if dist_utils.is_dist_available_and_initialized():
                current_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                current_dataloader,  # 선택된 데이터로더를 사용
                self.optimizer, 
                self.device, 
                epoch, 
                is_dsg_epoch=is_dsg_epoch, # is_dsg_epoch 플래그 전달
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

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

            # 평가 시에는 Dsg 데이터셋으로 평가를 진행합니다.
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader,    # Dsg 데이터셋을 사용하여 평가
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
