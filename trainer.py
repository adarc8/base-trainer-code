import gc
import glob
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import monai
import numpy as np
import torch
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm

from helpers.configuration.constants import DATA_FORMAT, RES, Output

from helpers.configuration.constants import Phase
from helpers.configuration.trainer_config import TrainerConfig
from helpers.functions import WaveletCompressor
from helpers.inferer import FlexibleConditionalDiffusionInfererCross
from helpers.metric_structe import Metric
from helpers.model import UnetWrapper
from helpers.prepare_batch import SamplePredictionPrepareBatch
from helpers.trainer_base import BaseTrainer


class Trainer(BaseTrainer, TrainerConfig):
    def __init__(self, config_dict: dict[str, any]):
        TrainerConfig.__init__(self, **config_dict)
        BaseTrainer.__init__(self)

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._dataloaders = self._define_dataloaders(self._batch_size, self._num_workers)
        self._model = UnetWrapper().to(self._device)
        self._opt = torch.optim.AdamW(self._model.parameters(), lr=self._learning_rate)

        self._define_diffusion_stuff()
        self._metrics = {"L2": Metric(loss_fn=torch.nn.MSELoss(), loss_weight=1)}


    def train(self):
        print(f"Starting training: {self._experiment_name}")
        epochs_tqdm = tqdm(range(self._start_epoch, self._end_epoch), initial=self._start_epoch, leave=True)
        for epoch in epochs_tqdm:
            epoch_start_time = time.time()
            for phase in Phase.values():
                # Run epoch
                epoch_metrics = self.run_phase_epoch(phase=phase, epoch=epoch)
                # Write metrics to tensorboard
                self._write_epoch_metrics_to_tb(epoch=epoch, writer=self._tb_writers[phase],
                                                epoch_metrics=epoch_metrics)
                self._save_checkpoint(phase, loss=epoch_metrics["loss"])

                epochs_tqdm.set_description(
                    f"Epoch: {epoch}, {phase} Loss:"
                    f"{epoch_metrics['loss']:.7f}, "
                    f"Epoch Time: {(time.time() - epoch_start_time):.3f} sec"
                )

    def _save_checkpoint(self, phase: Phase, loss: float):
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._opt.state_dict(),
        }
        if phase == Phase.VAL:
            if loss < self._model_min_loss:
                self._model_min_loss = loss
                checkpoint_path = self._output_dir_path[Output.CHECKPOINTS] / "best_checkpoint.pt"
                torch.save(checkpoint, checkpoint_path)
        elif phase == Phase.TRAIN:
            checkpoint_path = self._output_dir_path[Output.CHECKPOINTS] / f"last_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

    def run_phase_epoch(self, phase: Phase, epoch: int) -> dict[str, float]:
        is_training = phase == Phase.TRAIN
        self._set_model_mode(is_training, self._model)
        epoch_metrics = defaultdict(list)

        epoch_loss = 0.0
        for batch_idx, batch_data in enumerate(self._dataloaders[phase]):
            self._opt.zero_grad()
            gc.collect()
            with torch.set_grad_enabled(is_training), torch.cuda.amp.autocast(enabled=self._half_precision):
                images, target, _, info_dict = self._prepare_batch(batch_data, device=self._device, non_blocking=True)

                outputs = self._inferer(
                    inputs=images,
                    noise=info_dict["noise"],  # pure noise (torch.randn_like(images))
                    diffusion_model=self._model,
                    timesteps=info_dict["timesteps"],  # random between 0 and num_train_timesteps
                    conditioning=info_dict["conditioning"],  # the age given
                )

                loss = 0
                for metric_name, metric_fn in self._metrics.items():
                    with torch.set_grad_enabled(metric_fn.loss_weight != 0):  # Calc gradients only if loss_weight != 0
                        metric_val = metric_fn(outputs, target)
                        loss += metric_val.mean() * metric_fn.loss_weight
                        epoch_metrics[metric_name].append(metric_val.mean().item())

                epoch_metrics["loss"].append(loss.item())
                if is_training:
                    if self._half_precision:
                        self._scaler.scale(loss).backward()
                        self._scaler.step(self._opt)
                        self._scaler.update()
                    else:
                        loss.backward()
                        self._opt.step()

            if self._early_stop(epoch, batch_idx):
                break

        epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        return epoch_metrics


    def save_output_to_disk(self, epoch: Optional[int] = None):
        self._model.eval()
        # samples a new subject from the learned distribution at 7 different age values (55-80) using DDIM w/ 100 steps
        with torch.inference_mode():
            noise = torch.randn((1, 64, 48, 64, 48))
            noise = noise.to(self._device)
            self._scheduler.set_timesteps(num_inference_steps=1000)
            self._ddim_scheduler.set_timesteps(num_inference_steps=100)

            # age values predefined
            ages = [70.]
            conditioning = (torch.tensor(ages) - 46)
            conditioning = conditioning.unsqueeze(-1).unsqueeze(-1).to(self._device)
            with torch.cuda.amp.autocast():
                image = self._inferer.sample(input_noise=noise, diffusion_model=self._model,
                                             scheduler=self._ddim_scheduler,
                                             save_intermediates=False, conditioning=conditioning, mode='crossattn')
            image = image.cpu().numpy()
        for idx in range(noise.shape[0]):
            decoded_image = WaveletCompressor.decode(image[idx])
            age = ages[idx]
            slices = [100, 128, 156]
            for slice in slices:
                img_path = self._output_dir_path[Output.OUTPUT_IMAGES] / f"{slice=}" / f"{epoch=}.png"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_2d = decoded_image[0, :, slice]
                # transform to [0,1]
                img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-6)
                img_np8 = (img_2d * 255).astype(np.uint8)
                from PIL import Image
                img_pil = Image.fromarray(img_np8)
                img_pil.save(img_path)


    def _define_dataloaders(self, batch_size: int, num_workers: int) -> dict[str, monai.data.DataLoader]:
        data_loaders = {}
        for phase in Phase.values():
            # Read from disk
            image_list = glob.glob(DATA_FORMAT(phase))
            age_list = np.int32([70 for _ in range(len(image_list))])
            age_list = np.int32((age_list - 46) // 1.0)  # kind of normalization i guess
            files = [{"image": img, "age": label} for img, label in zip(image_list, age_list)]

            transforms = self._define_transforms()
            dataset = monai.data.Dataset(data=files, transform=transforms)

            shuffle = True if phase == Phase.TRAIN else False
            data_loader = monai.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=True)

            data_loaders[phase] = data_loader

        return data_loaders

    @staticmethod
    def _define_transforms():
        unsqueezer = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        percentile_info = {"lower": 1, "upper": 99, "b_min": -1, "b_max": 1, "clip": True}
        divisible_padding = 64

        transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True, reader="ITKReader"),
                monai.transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode='bilinear', lazy=True),
                monai.transforms.DivisiblePadd(keys=["image"], k=divisible_padding),
                monai.transforms.CenterSpatialCropd(keys=["image"], roi_size=RES),
                monai.transforms.ScaleIntensityRangePercentilesd(keys="image", **percentile_info),
                monai.transforms.Lambdad(keys=['image'], func=WaveletCompressor.encode),
                monai.transforms.ToTensord(keys=["image"], track_meta=False),
                monai.transforms.Lambdad(keys=["age"], func=unsqueezer),
            ],
            lazy=False)

        return transforms

    def _define_diffusion_stuff(self):
        self._scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")
        self._inferer = FlexibleConditionalDiffusionInfererCross(self._scheduler)
        self._prepare_batch = SamplePredictionPrepareBatch(num_train_timesteps=1000, condition_name='age')
        self._ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")


    def _early_stop(self, epoch: int, batch_idx: int) -> bool:
        """Will send stop singla if we have batch idx to stop, or if its first epoch, we stop quickly"""
        first_epoch_stop = (epoch == self._start_epoch and batch_idx >= self._first_epoch_batches)
        return (batch_idx >= self._batch_idx_to_stop_epoch) or first_epoch_stop
