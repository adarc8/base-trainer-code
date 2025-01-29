import argparse

from helpers.configuration.constants import DEBUG_PRINT, DEBUG_FIXED_CONFIG, DEBUG_EXPERIMENT_POSTFIX
from trainer import Trainer
from helpers.functions import read_yml_config, color_print


def main():
    args, config_dict = _read_config_dict()
    if args.debug:
        color_print(DEBUG_PRINT)
        config_dict['_experiment_name'] = f"{config_dict['_experiment_name']}{DEBUG_EXPERIMENT_POSTFIX}"
        config_dict.update(DEBUG_FIXED_CONFIG)

    trainer = Trainer(config_dict)
    trainer.train()


def _read_config_dict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='helpers/configuration/config.yaml', help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    config_dict = read_yml_config(args.config)
    return args, config_dict


if __name__ == '__main__':
    main()