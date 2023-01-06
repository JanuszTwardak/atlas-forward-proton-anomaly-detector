import itertools
import time
from typing import Dict, List, Optional, Union
import hydra
import matplotlib
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

# from visualize.DistributionPlots import DistributionPlots

log = logging.getLogger("visualize")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def visualize(cfg: DictConfig):
    """visualize Function to combine predicted scores with the rest of extracted data
    and to run all visualization plots.

    Args:
        cfg (DictConfig): config file.
    """

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
        scores = pd.read_csv(
            scores_paths[-1], names=["run_id", "evN", "score"]
        )
        scores["evN"] = scores["evN"].astype(str)
        data["evN"] = data["evN"].astype(str)
        data["score"] = data["score"].astype(float)
        log.info("Merging data with scores...")
        data = data.merge(right=scores, on="evN")
        data = data.dropna()

        with open(abspath(cfg.external.path), "wb") as f:
            pickle.dump(data, f)  # TODO: replace .pickle for .parquet
    else:
        with open(abspath(cfg.external.path), "rb") as f:
            data = pickle.load(f)

    if cfg.visualize.plots.boxplot.to_draw or cfg.draw_all_plots:

        Visualize.draw_score_violinplot(
            data=data,
            flierprops=cfg.visualize.flierprops,
            output_dir=cfg.reports.figures_dir,
        )

    if cfg.visualize.plots.grouped_violinplots.to_draw or cfg.draw_all_plots:

        Visualize.draw_grouped_violinplots(
            features=data.select_dtypes(include=np.number).columns,
            quart_features=["score"],
            df=data,
            flierprops=cfg.visualize.flierprops,
            output_dir=cfg.reports.figures_dir,
            axis_limits=cfg.visualize.plots.grouped_violinplots.axis_limits,
            range_limits=None,
        )

    if cfg.visualize.plots.pairplot.to_draw or cfg.draw_all_plots:
        Visualize.plot_pairplot(
            data=data,
            output_dir=cfg.reports.figures_dir,
            data_frac=cfg.visualize.taken_dataset_frac,
            kind=cfg.visualize.plots.pairplot.kind,
            cmap=cfg.visualize.cmap,
            pthresh=cfg.visualize.plots.pairplot.p_threshold,
            bins=cfg.visualize.plots.pairplot.bins,
            common_norm=cfg.visualize.plots.pairplot.use_common_norm,
            groups_limits=cfg.visualize.plots.pairplot.groups_limits,
            stat=cfg.visualize.plots.pairplot.stat,
        )


class Visualize:
    """
    Class containing all methods responsible for visualization.
    """

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
        stat: Optional[str] = "count",
        groups_limits: Optional[List[float]] = None,
    ) -> None:

        if data_frac < 1.0:
            data = data.sample(frac=data_frac)

        sns.set_theme(style="dark")

        full_data = Visualize.get_grouped(data, groups_limits)

        for group in full_data["score_quart"].unique():

            data = full_data[full_data["score_quart"] == group]

            # Visualize.draw_slopes(data, output_dir, group)

            p_grid = sns.PairGrid(data, height=2.5)
            time_start = time.time()
            log.info(f"Started pairplot plotting!")
            if kind == "histplot":
                logging.info(f"Plotting histplot on lower diagonal...")

                p_grid.map_lower(
                    sns.histplot,
                    bins=20,
                    pthresh=None,
                    hue=data["side"],
                    # cmap=cmap,
                    cbar=False,
                    stat=stat,
                )

                # loc = matplotlib.ticker.MultipleLocator(0.5)
                lims = [
                    (20, 80),
                    (3.5, 4.5),
                    (-15, -5),
                    (-15, -5),
                    (-20, 20),
                    (-20, 20),
                    (-0.002, 0.005),
                    (-0.02, 0.005),
                    (-0.01, 0.01),
                    (-0.02, 0.01),
                    (0, 300),
                    (0, 300),
                    (0, 75),  # hits_col_1
                    (0, 75),
                    (10000, 25000),
                    (0, 75),
                    (8.5, 9.5),
                    (0.3, 0.7),
                ]

                for ax, (ylims, xlims) in zip(
                    p_grid.axes.flat,
                    itertools.product(lims, lims),
                    # itertools.product(tick_inc, tick_inc)
                ):
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)

                p_grid.map_upper(
                    sns.kdeplot,
                    fill=True,
                )

                logging.info(f"Histplot on lower diagonal done!")

            p_grid.map_diag(plt.hist, bins=10)
            plt.legend(f"pairplot with score group in ({group})")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            plot_name = f"run3_pair_{kind}_{int(data_frac*100)}_{data_frac}_{group}.png"
            save_path = Path(output_dir) / plot_name
            plt.savefig(save_path)
            plt.clf()
            time_stop = time.time()
            log.info(
                f"Pairplot finished! Total time: {int((time_stop - time_start)/60)} minutes"
            )

    @staticmethod
    def draw_slopes(data: pd.DataFrame, output_dir: str, group_str: str):

        tracks_sx = {
            "tracks_sx_1": "tracks_sy_1",
            "tracks_sx_2": "tracks_sy_2",
        }
        plt.figure(figsize=(40, 20))
        plt.clf()
        plt.title(f"score in ({group_str})")
        i = 1
        for letter in ["a", "c"]:
            for tr in tracks_sx.items():
                plt.subplot(1, 4, i)

                plt.xlim((-0.025, 0.025))
                plt.ylim((-0.025, 0.025))
                dist = "NEAR" if tr[0][-1] == "1" else "FAR"
                plt.title(f"{letter.upper()} {dist} score=({group_str})")

                plt.ylabel("slope y")
                plt.xlabel("slope x")
                x = data[data["side"] == letter][tr[0]]
                y = data[data["side"] == letter][tr[1]]

                sns.kdeplot(data=data, x=x, y=y, fill=True, cmap="flare")
                i += 1

        plot_name = f"run3_slopes_grouped_{group_str}.png"
        save_path = Path(output_dir) / plot_name
        plt.savefig(str(save_path))
        plt.clf()

    @staticmethod
    def draw_score_violinplot(
        data: pd.DataFrame,
        flierprops: dict,
        output_dir: str,
    ) -> None:

        logging.info("Creating boxplot...")

        for version in [None, "side"]:
            sns.violinplot(
                x="score",
                y=version,
                data=data,
            )

            plt.title("Violinplot for all events")

            # quantiles = np.quantilfe(
            #     data["scores"], np.array([0.00, 0.25, 0.50, 0.75, 1.00])
            # )

            logging.info("Saving plot...")
            plot_name = f"run3_score_violinplot_all_{version}.png"
            save_path = Path(output_dir) / plot_name
            plt.savefig(str(save_path))
            logging.info("Plot saved")
            plt.clf()

    @staticmethod
    def get_grouped(
        df: pd.DataFrame,
        range_limits: Optional[Union[List[float], bool]] = None,
    ):
        grouping_feature = "score"

        if range_limits is None:
            range_limits = [0.0, 0.25, 0.5, 0.75, 1.0]
            range_name = "quartile"
            in_plot_title = "quartile"
            labels = ["1st", "2nd", "3rd", "4th"]
        else:
            range_name = ""
            for limit in range_limits:
                range_name += str(limit) + "_"
            range_name = range_name.replace(".", ",")
            labels = []
            for index, limit in enumerate(range_limits[:-1]):
                labels.append(f"{limit*100}%-{range_limits[index+1]*100}%")

        # df = df.sort_values(by=quart_feature)
        ranges = df[grouping_feature].quantile(range_limits).to_numpy()

        log.debug(ranges)
        df[f"{grouping_feature}_quart"] = pd.cut(
            df[grouping_feature],
            bins=ranges,
            labels=labels,
        )

        return df

    @staticmethod
    def draw_grouped_violinplots(
        features: List[str],
        quart_features: List[str],
        df: pd.DataFrame,
        flierprops: dict,
        output_dir: str,
        axis_limits: Dict[str, List[float]],
        range_limits: Optional[List[float]] = None,
    ) -> None:

        if range_limits is None:
            range_limits = [0.0, 0.25, 0.5, 0.75, 1.0]
            range_name = "quartile"
            in_plot_title = "quartile"
            labels = ["1st", "2nd", "3rd", "4th"]
        else:
            range_name = ""
            for limit in range_limits:
                range_name += str(limit) + "_"
            range_name = range_name.replace(".", ",")
            labels = []
            for index, limit in enumerate(range_limits[:-1]):
                labels.append(f"{limit*100}%-{range_limits[index+1]*100}%")

        for quart_feature in quart_features:
            # df = df.sort_values(by=quart_feature)
            ranges = df[quart_feature].quantile(range_limits).to_numpy()

            log.debug(ranges)
            df[f"{quart_feature}_quart"] = pd.cut(
                df[quart_feature],
                bins=ranges,
                labels=labels,
            )

            for feature in features:
                plot = sns.violinplot(
                    data=df,
                    x=feature,
                    y=f"{quart_feature}_quart",
                    hue="side",
                    split=True,
                )
                plt.title(
                    f"{feature} violinplot grouped by {quart_feature} groups"
                )
                plt.ylabel(f"{quart_feature} group")
                plt.xlabel(f"{feature}")
                plot_name = f"{feature}_groupedby_{quart_feature}_groups_violinplot.png"
                save_dir = Path(output_dir) / f"violinplots_{range_name}"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / plot_name
                plt.savefig(save_path)
                logging.info("Plot saved")
                plt.clf()

                if feature in axis_limits:

                    plot = sns.violinplot(
                        data=df,
                        x=feature,
                        y=f"{quart_feature}_quart",
                        hue="side",
                        split=True,
                    )
                    plt.title(
                        f"{feature} violinplot grouped by {quart_feature} groups"
                    )
                    plt.ylabel(f"{quart_feature} group")
                    plt.xlabel(f"{feature}")
                    plot.set(xlim=(tuple(axis_limits[feature])))
                    plot_name = f"{feature}_groupedby_{quart_feature}_quartiles_violinplot_limited.png"
                    save_dir = Path(output_dir) / f"violinplots_{range_name}"
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / plot_name
                    plt.savefig(save_path)
                    logging.info("Plot saved")
                    plt.clf()


if __name__ == "__main__":
    visualize()
