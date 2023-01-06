import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path as abspath
import logging
from process.RootHandler import RootHandler

log = logging.getLogger("process")


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def process_data(cfg: DictConfig) -> None:
    """process_data Function to run all preprocessing functions with options specified in config.

    Args:
        cfg (DictConfig): Config file.
    """

    RootHandler.extract_root(
        root_paths=[abspath(path) for path in cfg.raw.paths],
        branches_to_extract=cfg.process.branches_to_extract,
        chunk_size=cfg.process.memory_chunk_size,
        hits_n_limits=tuple(cfg.process.hits_n_limits),
        hits_tracks_limits=tuple(cfg.process.hits_tracks_limits),
        events_limit_no=cfg.process.events_limit_no,
        output_dir=abspath(cfg.processed.dir),
    )


if __name__ == "__main__":
    process_data()
