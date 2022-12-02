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
    """Function to process the data"""

    print(f"Functions used: {cfg.process.use_functions}")

    RootHandler.extract_root(
        root_paths=[abspath(path) for path in cfg.raw.paths],
        branches_to_extract=cfg.process.branches_to_extract,
        chunk_size=cfg.process.memory_chunk_size,
        min_hits_no=cfg.process.min_hits_number,
        max_hits_no=cfg.process.max_hits_number,
        events_limit_no=cfg.process.events_limit_no,
        output_dir=abspath(cfg.processed.dir),
    )


if __name__ == "__main__":
    process_data()
