import pickle
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath
from isotree import IsolationForest
import pandas as pd
from pathlib2 import Path
from tqdm import tqdm
from datetime import datetime


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train_model(cfg: DictConfig) -> None:
    """Function to train the model on all data from processed dir. It should have .parquet format."""

    print(f"Functions used: {cfg.model.name}")

    # input_path = abspath(config.processed.path)
    # output_path = abspath(config.final.path)

    # print(f"Train modeling using {input_path}")
    # print(f"Model used: {config.model.name}")
    # print(f"Save the output to {output_path}")

    model = IsolationForest(
        ndim=cfg.model.forest_dimensions,
        ntrees=cfg.model.ntrees,
    )

    dataset_path_list = [
        path
        for path in Path(abspath(cfg.processed.dir)).iterdir()
        if path.suffix == ".parquet"
    ]

    for chunk_path in tqdm(dataset_path_list):
        chunk = pd.read_parquet(chunk_path, engine="pyarrow")
        model = model.partial_fit(chunk.drop(columns=["evN", "run_id"]))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_save_path = Path(abspath(cfg.model_dir)) / timestamp

    with open(str(model_save_path), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
