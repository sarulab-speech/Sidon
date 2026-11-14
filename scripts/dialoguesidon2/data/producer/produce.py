"""Write one producer task's shards: <output_root>/<writer_name>/train/t<NN>/ and, for task 0 only, valid/t00/.

Task NN takes part NN of the training source shards (np.array_split of the shard listing into n_shards parts); the
validation shards are written once, by task 0. Records are written by `sidon.preprocess.process_item` (uuid4 keys)
through webdataset ShardWriters (worker-<n>-dataset-<NNNNNN>.tar; a shard closes at 3 GB, about 31 records).

  python scripts/dialoguesidon2/data/producer/produce.py data=dialoguesidon2_producer \\
      "data.datamodule.train_wds_patterns=[SOURCE/train]" "data.datamodule.val_wds_patterns=[SOURCE/dev]" \\
      "data.datamodule.noise_path=[NOISE]" data.datamodule.shard_idx=0 \\
      preprocess.output_root=OUT preprocess.writer_name=openyap_24k_120s hydra.run.dir=LOGS/t00

PRODUCER_TASK_NAME overrides the task directory name (default t<shard_idx, 2 digits>).
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from sidon.preprocess import run_parallel_writing  # noqa: E402


def task_name(shard_idx: int) -> str:
    return os.environ.get("PRODUCER_TASK_NAME") or f"t{int(shard_idx):02d}"


@hydra.main(config_path="../../../../config", config_name="preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup()
    name = task_name(datamodule.shard_idx)
    output_root = pathlib.Path(cfg.preprocess.output_root) / cfg.preprocess.writer_name
    if datamodule.shard_idx == 0:
        run_parallel_writing(datamodule.val_dataloader(), output_root / "valid" / name, cfg.preprocess.val_num_writers,
                             cfg.preprocess.shard_maxcount, cfg.preprocess.queue_size_per_worker)
    run_parallel_writing(datamodule.train_dataloader(), output_root / "train" / name, cfg.preprocess.train_num_writers,
                         cfg.preprocess.shard_maxcount, cfg.preprocess.queue_size_per_worker)


if __name__ == "__main__":
    main()
