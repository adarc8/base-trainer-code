from dataclasses import dataclass, asdict
from typing import Any, List

# DATA_FORMAT = lambda phase: f'../../data/brats23_{phase}/*/*t1c.nii.gz'
DATA_FORMAT = lambda phase: f'../../data/subset_{phase}/*/*t1c.nii.gz'
RES = [192, 256, 192]

# Debugging
DEBUG_PRINT = " ~~~~~~ @@@@ ~~~~~~ Debugging mode ~~~~~~ @@@@ ~~~~~~ "
DEBUG_FIXED_CONFIG = {
            '_batch_idx_to_stop_epoch': 1,
            '_batch_size': 1,
            '_num_workers': 0,
            '_epochs': 2,
        }
DEBUG_EXPERIMENT_POSTFIX = "_@@@@@@@_DELETE_THIS_ITS_DEBUGGING_@@@@@@@"



TRAIN = 'train'
VAL = 'val'

CHECKPOINTS_DIR_NAME = 'checkpoints'
TENSORBOARD_DIR_NAME = 'tensorboards'
OUTPUT_IMAGES_DIR_NAME = 'output_images'
OUTPUT_CODE_DIR_NAME = 'output_code'


@dataclass
class DataclassWithReturnAllValues:
    """This dataclass has the ability to return all its values as a list."""

    @classmethod
    def values(cls) -> List[Any]:
        return list(asdict(cls()).values())  # Create an instance of the class to get its values


@dataclass
class Phase(DataclassWithReturnAllValues):
    """This is kind of like the 'Config' of the different phases (train/val)"""
    TRAIN: str = TRAIN
    VAL: str = VAL


@dataclass
class Output(DataclassWithReturnAllValues):
    """This is kind of like the 'Config' of the different output directories we work with"""
    CHECKPOINTS: str = CHECKPOINTS_DIR_NAME
    TENSORBOARD: str = TENSORBOARD_DIR_NAME
    OUTPUT_IMAGES: str = OUTPUT_IMAGES_DIR_NAME
    OUTPUT_CODE: str = OUTPUT_CODE_DIR_NAME
