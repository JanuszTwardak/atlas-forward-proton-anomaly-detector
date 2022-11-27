import pickle
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath
from isotree import IsolationForest
import pandas as pd
from pathlib2 import Path
from tqdm import tqdm
from datetime import datetime
import logging

log = logging.getLogger("__name__")


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
    if cfg.to_train:
        _train_or_load(
            forest_dimensions=cfg.model.forest_dimensions,
            ntrees=cfg.model.ntrees,
            input_data_dir=abspath(cfg.processed.dir),
            model_dir=abspath(cfg.model_dir),
            use_tracks=cfg.model.use_tracks,
        )

    _predict_scores(
        model_dir=abspath(cfg.model_dir),
        data_dir=abspath(cfg.processed.dir),
        reports_dir=abspath(cfg.reports_dir),
        use_tracks=cfg.model.use_tracks,
    )


def _train_or_load(
    forest_dimensions: int,
    ntrees: int,
    input_data_dir: str,
    model_dir: str,
    use_tracks: bool,
) -> None:

    model = IsolationForest(
        ndim=forest_dimensions,
        ntrees=ntrees,
    )

    dataset_path_list = [
        path
        for path in Path(input_data_dir).iterdir()
        if path.suffix == ".parquet"
    ]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log.info(f"Training model {timestamp}")
    for chunk_path in tqdm(dataset_path_list):
        chunk = pd.read_parquet(chunk_path, engine="pyarrow")
        _learning_chunk = chunk.drop(columns=["evN", "run_id"])
        if not use_tracks:
            _learning_chunk = _learning_chunk.drop(
                _learning_chunk.filter(regex="^tracks_", axis=1), axis=1
            )
        model = model.partial_fit(_learning_chunk)

    model_save_path = Path(abspath(model_dir)) / f"{timestamp}.pickle"

    with open(str(model_save_path), "wb") as f:
        pickle.dump(model, f)


def _predict_scores(
    model_dir: str, data_dir: str, reports_dir: str, use_tracks: bool
) -> None:
    """predict_scores Using already trained model predict dataframe anomaly scores. They will be saved in .csv file.

    Parameters
    ----------
    model : IsolationForest
        Fully trained IsolationForest model.
    preprocessed_df_names : List[str]
        List of dataframe names (that were used to train model) to predict score of.
    preprocessed_df_dict_path : Optional[List[str]], optional
        Path to folder containing training dataframes, by default parameters.preprocessed_data_path
    final_results_path : Optional[str], optional
        Path to folder containing final results of model data, by default parameters.final_results_path
    """

    model_paths = sorted(
        [
            str(path)
            for path in Path(abspath(model_dir)).iterdir()
            if path.suffix == ".pickle"
        ]
    )

    with open(str(model_paths[-1]), "rb") as f:
        model = pickle.load(f)

    scores_save_path = Path(reports_dir) / f"{Path(model_paths[-1]).stem}.csv"

    with open(str(scores_save_path), "w+") as f:
        pass

    dataset_path_list = [
        path for path in Path(data_dir).iterdir() if path.suffix == ".parquet"
    ]

    log.info(f"Predicting scores using model {Path(model_paths[-1]).stem}...")
    for chunk_path in tqdm(dataset_path_list):
        chunk = pd.read_parquet(chunk_path, engine="pyarrow")
        _pred_chunk = chunk.drop(columns=["evN", "run_id"])
        if not use_tracks:
            _pred_chunk = _pred_chunk.drop(
                _pred_chunk.filter(regex="^tracks_", axis=1), axis=1
            )
        scores = chunk[["run_id", "evN"]].copy()

        scores["score"] = pd.DataFrame(
            model.predict(_pred_chunk),
        )

        scores.to_csv(scores_save_path, mode="a", index=False, header=True)


if __name__ == "__main__":
    train_model()
