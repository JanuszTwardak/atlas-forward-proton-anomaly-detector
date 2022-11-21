import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath


@hydra.main(config_path="../config", config_name="main")
def visualize(config: DictConfig):
    """Function to train the model"""
    pass
    # input_path = abspath(config.final.path)
    # results_dir = abspath(config.reports.dir)
    # figures_dir = abspath(config.reports.figures_dir)

    # print(f"Visualize using {input_path}")
    # print(f"Save the output to {results_dir} and {figures_dir}")


if __name__ == "__main__":
    visualize()
