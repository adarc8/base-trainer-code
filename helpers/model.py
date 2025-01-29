from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
import torch


class UnetWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            num_channels=[128, 128, 256, 256, 512],
            attention_levels=[False, False, False, True, True],
            num_head_channels=[0, 0, 0, 32, 32],
            num_res_blocks=2,
            use_flash_attention=False,
            with_conditioning=True,
            cross_attention_dim=1
        )

    def forward(self, x, timesteps, context):
        return self.unet(x, timesteps, context=context)

