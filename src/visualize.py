import time
from typing import List, Optional
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath
from pathlib2 import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import logging
from matplotlib import pyplot as plt
import pickle

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def visualize(cfg: DictConfig):
    """Function to train the model"""

    # input_path = abspath(config.final.path)
    # results_dir = abspath(config.reports.dir)
    # figures_dir = abspath(config.reports.figures_dir)

    # print(f"Visualize using {input_path}")
    # print(f"Save the output to {results_dir} and {figures_dir}")

    if cfg.to_merge_scores_and_data:
        data_chunks_path_list = [
            path
            for path in Path(abspath(cfg.processed.dir)).iterdir()
            if path.suffix == ".parquet"
        ]

        data = pd.concat(
            map(pd.read_parquet, data_chunks_path_list), ignore_index=True
        )

        scores_paths = sorted(
            [
                str(path)
                for path in Path(abspath(cfg.reports.dir)).iterdir()
                if path.suffix == ".csv"
            ]
        )
        log.info(f"Using {Path(scores_paths[-1]).stem}")
        scores = pd.read_csv(scores_paths[-1])

        scores["evN"] = scores["evN"].astype(str)
        data["evN"] = data["evN"].astype(str)
        data = data.merge(right=scores, on="evN")

        data = data.dropna()
        data["score"] = data["score"].astype(float)

        with open(abspath(cfg.external.path), "wb") as f:
            pickle.dump(data, f)  # TODO: replace .pickle for .parquet
    else:
        with open(abspath(cfg.external.path), "rb") as f:
            data = pickle.load(f)

    if cfg.visualize.plots.pairplot.to_draw or cfg.draw_all_plots:
        Visualize.plot_pairplot(
            data=data,
            output_dir=cfg.figures_dir,
            data_frac=cfg.visualize.taken_dataset_frac,
            kind=cfg.visualize.plots.pairplot.kind,
            cmap=cfg.visualize.cmap,
            pthresh=cfg.visualize.plots.pairplot.p_threshold,
            bins=cfg.visualize.plots.pairplot.bins,
            common_norm=cfg.visualize.plots.pairplot.use_common_norm,
        )

    if cfg.visualize.plots.boxplot.to_draw or cfg.draw_all_plots:
        Visualize.draw_score_boxplots(
            data=data,
            flierprops=cfg.visualize.flierprops,
            output_dir=cfg.figures_dir,
        )

    if cfg.visualize.plots.quartile_boxplot.to_draw or cfg.draw_all_plots:
        Visualize.draw_quartile_boxplots(
            features=[
                "hits_n",
                "tracks",
                "tracks_x_1",
                "tracks_x_2",
                "tracks_sx_1",
                "tracks_sx_2",
            ],
            data=data,
            flierprops=cfg.visualize.flierprops,
            output_dir=cfg.figures_dir,
        )


class Visualize:
    @staticmethod
    def plot_pairplot(
        data: pd.DataFrame,
        output_dir: str,
        data_frac: float,
        kind: str,
        cmap: Optional[str] = None,
        pthresh: Optional[float] = None,
        bins: Optional[int] = 2000,
        common_norm: Optional[bool] = False,
    ) -> None:

        if data_frac < 1.0:
            data = data.sample(frac=data_frac)

        sns.set_theme(style="dark")

        p_grid = sns.PairGrid(data, height=2.5)
        time_start = time.time()
        log.info(f"Started pairplot plotting!")
        if kind == "histplot":
            logging.info(f"Plotting histplot on lower diagonal...")

            p_grid.map_lower(
                sns.histplot,
                bins=bins,
                pthresh=pthresh,
                cmap=cmap,
                cbar=False,
            )

            logging.info(f"Histplot on lower diagonal done!")

        p_grid.map_diag(plt.hist, bins=20, color="orange")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plot_name = f"run3_pair_{kind}_{int(data_frac*100)}_{data_frac}.png"
        save_path = Path(output_dir) / plot_name
        plt.savefig(save_path)
        plt.clf()
        time_stop = time.time()
        log.info(
            f"Pairplot finished! Total time: {int((time_stop - time_start)/60)} minutes"
        )

    @staticmethod
    def draw_score_boxplots(
        data: pd.DataFrame,
        flierprops: dict,
        output_dir: str,
    ) -> None:

        logging.info("Creating boxplot...")

        for version in [None, "side"]:
            sns.boxplot(
                x="score",
                y=version,
                data=data,
                flierprops=flierprops,
            )

            plt.title("Scores boxplot for all events")

            # quantiles = np.quantilfe(
            #     data["scores"], np.array([0.00, 0.25, 0.50, 0.75, 1.00])
            # )

            logging.info("Saving plot...")
            plot_name = f"run3_score_boxplot_all_{version}.png"
            save_path = Path(output_dir) / plot_name
            plt.savefig(str(save_path))
            logging.info("Plot saved")
            plt.clf()

    @staticmethod
    def draw_quartile_boxplots(
        features: List[str],
        data: pd.DataFrame,
        flierprops: dict,
        output_dir: str,
    ) -> None:

        quartile_extremes = data["score"].quantile(
            q=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        quartile_extremes = quartile_extremes.to_numpy()

        for feature in features:
            quartiles = []
            for i in range(1, 5):
                _temp = data[
                    (data["score"] >= quartile_extremes[i - 1])
                    & (data["score"] <= quartile_extremes[i])
                ]
                quartiles.append(_temp[feature].values.tolist())

            plot = sns.boxplot(data=quartiles, flierprops=flierprops)
            plot.set_xticklabels([f"Q{i}" for i in range(1, 5)])
            plt.title(f"{feature} boxplots for anomaly score quartiles")
            plt.xlabel("Anomaly score quartile")
            plt.ylabel(f"{feature}")
            plot_name = f"{feature}_quantile_boxplot.png"
            save_path = Path(output_dir) / plot_name
            plt.savefig(save_path)
            logging.info("Plot saved")
            plt.clf()


if __name__ == "__main__":
    visualize()
