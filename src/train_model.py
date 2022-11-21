import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath


@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):
    """Function to train the model"""
    pass
    # input_path = abspath(config.processed.path)
    # output_path = abspath(config.final.path)

    # print(f"Train modeling using {input_path}")
    # print(f"Model used: {config.model.name}")
    # print(f"Save the output to {output_path}")


if __name__ == "__main__":
    train_model()
