import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Tuple

import torch


@dataclass
class TrainerConfig:
    _seed: int  # random seed
    _start_epoch: int  # manual epoch number (useful on restarts)
    _epochs: int  # Number of epochs to train.
    _batch_size: int  # Number of epochs to train.
    _learning_rate: float  # The learning rate.
    _data_path: str  # Path to where your profiles sit at
    _experiment_name: str  # give name to this experiment training process
    _num_workers: int  # Number of workers for data_preprocess loading
    _pretrained_checkpoint_path: Optional[str]  # Path to pretrained checkpoint (None if not using)
    _output_base_path: str  # the base outputs for all experiments (recommend "output")
    _batch_idx_to_stop_epoch: Union[int, float]  # stop epoch when batch_idx reach this number or when epoch ends
    _half_precision: bool  # If True, will use half precision
    _paths_to_save: List[str]  # Paths to save the model
    _first_epoch_batches: int  # Number of batches to run on first epoch (recommend 1 batch only)

    def __post_init__(self):
        """This func helps to handle the config parameters."""

        self._data_path = Path(self._data_path)
        self._output_base_path = Path(self._output_base_path)
        self._paths_to_save = [Path(p) for p in self._paths_to_save]
        self._experiment_name = self._add_time_to_experiment_name(self._experiment_name)

        self._learning_rate = float(self._learning_rate)

        if self._batch_idx_to_stop_epoch is None:
            self._batch_idx_to_stop_epoch = float('inf')
        self._end_epoch = self._start_epoch + self._epochs

        self._model_min_loss = float('inf')

    @staticmethod
    def _add_time_to_experiment_name(experiment_name: str) -> str:
        """Add time to experiment name"""
        time_now = f"_{datetime.datetime.now():%H_%M_%S}"
        return f"{experiment_name}{time_now}"

