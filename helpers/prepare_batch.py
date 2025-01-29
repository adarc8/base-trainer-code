from typing import Optional

from generative.engines import DiffusionPrepareBatch


class SamplePredictionPrepareBatch(DiffusionPrepareBatch):
    """
    This class is used as a callable for the `prepare_batch` parameter of engine classes for diffusion training.
    """

    def __init__(self, num_train_timesteps: int, condition_name: Optional[str] = None) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, condition_name=condition_name)

    def get_target(self, images, noise, timesteps):
        return images