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

log = logging.getLogger("visualize")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def visualize(cfg: DictConfig):
    """Function to train the model"""

    scores_paths = sorted(
        [
            str(path)
            for path in Path(abspath(cfg.reports.dir)).iterdir()
            if path.suffix == ".csv"
        ]
    )

    if cfg.to_merge_scores_and_data:
        log.info("Loading processed data...")
        data_chunks_path_list = [
            path
            for path in Path(abspath(cfg.processed.dir)).iterdir()
            if path.suffix == ".parquet"
        ]

        data = pd.concat(
            map(pd.read_parquet, data_chunks_path_list), ignore_index=True
        )
        log.info("Done!")
        log.info(f"Using {Path(scores_paths[-1]).stem}")
        log.info("Loading scores...")
        scores = pd.read_csv(
            scores_paths[-1], names=["run_id", "evN", "score"]
        )
        log.info("Done!")
        scores["evN"] = scores["evN"].astype(str)
        data["evN"] = data["evN"].astype(str)
        log.info("Merging data with scores...")
        data = data.merge(right=scores, on="evN")
        log.info("Done!")
        # data = data.dropna()
        data["score"] = data["score"].astype(float)

        with open(abspath(cfg.external.path), "wb") as f:
            pickle.dump(data, f)  # TODO: replace .pickle for .parquet
    else:
        with open(abspath(cfg.external.path), "rb") as f:
            data = pickle.load(f)

    print(data.info())
    print(data.describe())

    # plt.title(f"Anomaly score boxplot")
    # plot_name = f"{Path(scores_paths[-1]).stem}_boxplot.png"
    # save_dir = Path(cfg.figures_dir)
    # save_path = save_dir / plot_name
    # plt.savefig(save_path)
    # logging.info("Model boxplot saved")
    # plt.clf()

    # if cfg.visualize.plots.boxplot.to_draw or cfg.draw_all_plots:
    #     Visualize.draw_score_boxplots(
    #         data=data,
    #         flierprops=cfg.visualize.flierprops,
    #         output_dir=cfg.figures_dir,
    #     )

    # if cfg.visualize.plots.quartile_boxplot.to_draw or cfg.draw_all_plots:
    #     Visualize.draw_quartile_boxplots(
    #         features=data.select_dtypes(include=np.number).columns,
    #         quart_features=["score"],
    #         df=data,
    #         flierprops=cfg.visualize.flierprops,
    #         output_dir=cfg.figures_dir,
    #     )

    # if cfg.visualize.plots.pairplot.to_draw or cfg.draw_all_plots:
    #     Visualize.plot_pairplot(
    #         data=data,
    #         output_dir=cfg.figures_dir,
    #         data_frac=cfg.visualize.taken_dataset_frac,
    #         kind=cfg.visualize.plots.pairplot.kind,
    #         cmap=cfg.visualize.cmap,
    #         pthresh=cfg.visualize.plots.pairplot.p_threshold,
    #         bins=cfg.visualize.plots.pairplot.bins,
    #         common_norm=cfg.visualize.plots.pairplot.use_common_norm,
    #     )


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
                # cmap=cmap,
                cbar=False,
            )

            logging.info(f"Histplot on lower diagonal done!")

        # p_grid.map_diag(plt.hist, bins=20, color="orange")

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
        quart_features: List[str],
        df: pd.DataFrame,
        flierprops: dict,
        output_dir: str,
    ) -> None:

        for quart_feature in quart_features:
            df = df.sort_values(by=quart_feature)
            quartile_ranges = (
                df[quart_feature]
                .quantile([0.0, 0.25, 0.5, 0.75, 1.0])
                .to_numpy()
            )

            log.debug(quartile_ranges)

            df[f"{quart_feature}_quart"] = pd.cut(
                df[quart_feature],
                bins=quartile_ranges,
                labels=["1st", "2nd", "3rd", "4th"],
            )

            for feature in features:
                plot = sns.boxplot(
                    data=df,
                    x=feature,
                    y=f"{quart_feature}_quart",
                    flierprops=flierprops,
                )
                plt.title(
                    f"{feature} boxplots grouped by {quart_feature} quartile "
                )
                plt.ylabel(f"{quart_feature} quartile")
                plt.xlabel(f"{feature}")
                plot_name = f"{feature}_groupedby_{quart_feature}_quartiles_boxplot.png"
                save_dir = Path(output_dir) / "quartile_boxplots"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / plot_name
                plt.savefig(save_path)
                logging.info("Plot saved")
                plt.clf()


if __name__ == "__main__":
    visualize()
