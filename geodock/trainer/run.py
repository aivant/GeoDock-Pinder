import hydra
import torch
import os
from pathlib import Path

from omegaconf import DictConfig

_root = Path(__file__).parent
_confdir  = (_root / "config").as_posix()
@hydra.main(version_base="1.3", config_path=_confdir, config_name="config.yaml")
def main(config: DictConfig):
    torch.manual_seed(0)

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from geodock.trainer.train import train
    from geodock.trainer import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
