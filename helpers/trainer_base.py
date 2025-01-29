import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from helpers.configuration.constants import Output, Phase

from abc import ABC, abstractmethod



class BaseTrainer(ABC):
    """
    This is a base class to store basic trainer functions we always use.
    There is a recommend .train() func to use at the end of this class.

    This class assumes to have some attributes such as self._seed, self._experiment_name, etc.
    Therefore, when inheriting this class, make sure to define these attributes in the child class.
    Lastly, make sure you need to implement the _save_checkpoint method in the child class.
    """

    @abstractmethod
    def _save_checkpoint(self):
        """This method must be implemented by any subclass."""
        pass

    def __init__(self):
        """
        This Base class expect to have the following attributes in the child class:
        self._seed: int
        self._experiment_name: str
        self._output_base_path: str # Base path to save the output
        self._paths_to_save: list[Path] # List of paths to save in the output dir
        """
        self._seed_everything(self._seed)
        self._output_dir_path = self._define_output_dirs(self._output_base_path, self._experiment_name)
        self._save_important_files_in_output(self._output_dir_path[Output.OUTPUT_CODE], self._paths_to_save)
        self._tb_writers = self._define_phase_tb_writers_dict(self._output_dir_path)

    @staticmethod
    def _seed_everything(seed: int):
        """This func seeds everything"""
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # NumPy
        np.random.seed(seed)
        # Python's built-in random module
        random.seed(seed)

    @staticmethod
    def _save_important_files_in_output(output_code_dir_path: Path, paths_to_save: list[Path]):
        """This func saves important files in the output dir"""
        for path_to_save in paths_to_save:
            if path_to_save.is_dir():
                dest_path = output_code_dir_path / path_to_save.name
                shutil.copytree(path_to_save, dest_path)
            else:
                shutil.copy(path_to_save, output_code_dir_path)

    @staticmethod
    def _define_output_dirs(output_base_path: str, experiment_name: str) -> dict[Output, Path]:
        """Defines output_dir_path for every output dir, and also creates the dirs if they don't exist"""
        output_path = {}
        all_dirs: list[Output] = Output.values()
        for dir_name in all_dirs:
            output_path[dir_name] = Path(output_base_path) / experiment_name / dir_name
            output_path[dir_name].mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _define_phase_tb_writers_dict(output_path: dict[str, Path],
                                      tb_name: str = "") -> dict[Phase, SummaryWriter]:
        """
        Define model output tensorboard writers for each phase.
        tb_name is Optional postfix for the writer name (default is empty string).
        """
        phase_tb_writers_dict = {}
        for phase in Phase.values():
            writer_phase_path = output_path[Output.TENSORBOARD] / f"{phase}{tb_name}"
            phase_tb_writers_dict[phase] = SummaryWriter(writer_phase_path)
        return phase_tb_writers_dict

    @staticmethod
    def _remove_dir_if_exist(checkpoints_dir_path: Path):
        if checkpoints_dir_path.exists():
            shutil.rmtree(checkpoints_dir_path)

    @staticmethod
    def _set_model_mode(is_training: bool, model: torch.nn.Module) -> None:
        if is_training:
            model.train()
        else:
            model.eval()

    @staticmethod
    def _write_epoch_metrics_to_tb(epoch: int, writer: SummaryWriter,
                                   epoch_metrics: dict[str, float]):
        """Write metrics output to tb writer (for each phase)"""
        for metric_name, metric_val in epoch_metrics.items():
            writer.add_scalar(f'{metric_name}_epoch', metric_val, epoch)




"""
    Recommend train func
    def train(self):
    print(f"Starting training: {self._experiment_name}")
    epochs_tqdm = tqdm(range(self._start_epoch, self._end_epoch), initial=self._start_epoch, leave=True)
    for epoch in epochs_tqdm:
        epoch_start_time = time.time()
        phase_loss = {}
        for phase in Phase.values():
            # Run epoch
            epoch_metrics = self._run_phase_epoch(phase=phase, epoch=epoch)
            # Write metrics to tensorboard
            self._write_epoch_metrics_to_tb(epoch=epoch, writer=self._tb_writers[phase],
                                            epoch_metrics=epoch_metrics)
            # Store loss for printing & checkpointing
            phase_loss[phase] = epoch_metrics["loss"]

        self._save_checkpoint(epoch, val_loss=phase_loss[Phase.VAL])

        epochs_tqdm.set_description(
            f"Epoch: {epoch}, Train/Val Loss:"
            f"{phase_loss[Phase.TRAIN]:.7f}/"
            f"{phase_loss[Phase.VAL]:.7f}, "
            f"Epoch Time: {(time.time() - epoch_start_time):.3f} sec"
        )
    """