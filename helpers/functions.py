# todo change it to utils once u remove all previous utils
import numpy as np
import pywt
import torch
import yaml

from helpers.configuration.constants import RES

levels = 2  # Copilot: the number of wavelet decomposition levels


class WaveletCompressor:
    """Utility class for encode and decode with wavelet compression"""

    @staticmethod
    def encode(x):
        """Take image x and return a compressed wavelet representation"""
        packet = pywt.WaveletPacketND(x.get_array(), 'haar', axes=(-3, -2, -1), maxlevel=levels)
        x.set_array(torch.tensor(np.concatenate([y.data.view() for y in packet.get_level(levels)], axis=0) / 8.))
        return x


    @staticmethod
    def decode(x):
        """Take a compressed wavelet representation and return the original image"""
        res = [1] + RES
        dwt_packet = pywt.WaveletPacketND(np.zeros(res), 'haar', axes=(1, 2, 3), maxlevel=levels)
        dwt_list = dwt_packet.get_level(levels)


        for i, node in enumerate(dwt_list):
            dwt_packet[node.path].data = x[i, :, :, :][np.newaxis, :, :, :] * 8.
        x_hat = dwt_packet.reconstruct()

        return x_hat


def color_print(string):
    """Prints a string in a color"""
    print(f"\033[91m {string}\033[00m")


def read_yml_config(yaml_file: str) -> dict[str, any]:
    """This func loads the config from yaml file"""
    with open(yaml_file, 'r') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    if '_learning_rate' in config_dict:
        # when writing learning rate with "e" it gives string type
        config_dict['_learning_rate'] = float(config_dict['_learning_rate'])  # todo dont i deal with it already?
    return config_dict

