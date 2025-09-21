"""Materialise preprocessed WebDataset shards from the WebDataset data module."""

from __future__ import annotations

import os
import pathlib
import uuid
from multiprocessing import Manager, Process, Queue
from typing import Any, Dict

import hydra
import torch
import tqdm
import webdataset as wds
from lightning.pytorch import seed_everything
from omegaconf import DictConfig

from sidon.data.preprocess import WebDatasetDataModule


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datamodule samples into WebDataset-compatible records."""
    output_item: Dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            output_item[f"{key}.pth"] = value
        elif isinstance(value, str):
            output_item[f"{key}.txt"] = value
        elif isinstance(value, int):
            output_item[f"{key}.index"] = value
        else:
            output_item[f"{key}.pickle"] = value
    output_item["__key__"] = uuid.uuid4().hex
    return output_item


def writer_process(
    worker_id: int,
    queue: Queue,
    shard_maxcount: int,
    output_dir: pathlib.Path,
) -> None:
    """Drain items from the queue and write them into shard files."""
    shard_pattern = output_dir / f"worker-{worker_id}-dataset-%06d.tar"
    with wds.ShardWriter(str(shard_pattern), maxcount=shard_maxcount) as sink:
        while True:
            item = queue.get()
            if item is None:
                break
            sink.write(process_item(item))


def run_parallel_writing(
    dataloader: torch.utils.data.DataLoader,
    output_dir: pathlib.Path,
    num_writers: int,
    shard_maxcount: int,
    queue_size_per_worker: int,
) -> None:
    """Spawn writer processes that flush the dataloader contents to shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if num_writers <= 0:
        raise ValueError("num_writers must be positive")
    with Manager() as manager:
        data_queue: Queue = manager.Queue(maxsize=max(1, num_writers * queue_size_per_worker))
        processes: list[Process] = []
        for worker_id in range(num_writers):
            process = Process(
                target=writer_process,
                args=(worker_id, data_queue, shard_maxcount, output_dir),
            )
            process.start()
            processes.append(process)

        for item in tqdm.tqdm(dataloader, desc=f"Writing to {output_dir}"):
            data_queue.put(item)

        for _ in range(num_writers):
            data_queue.put(None)

        for process in processes:
            process.join()


def instantiate_datamodule(cfg: DictConfig) -> WebDatasetDataModule:
    datamodule: WebDatasetDataModule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup()
    return datamodule


@hydra.main(config_path="../../config", config_name="preprocess", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for generating preprocessed WebDataset shards."""
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)

    datamodule = instantiate_datamodule(cfg)
    job_id = os.getenv("PBS_JOBID", "local_run")
    output_root = pathlib.Path(cfg.preprocess.output_root) / cfg.preprocess.writer_name

    run_parallel_writing(
        datamodule.val_dataloader(),
        output_root / "valid" / job_id,
        cfg.preprocess.val_num_writers,
        cfg.preprocess.shard_maxcount,
        cfg.preprocess.queue_size_per_worker,
    )
    run_parallel_writing(
        datamodule.train_dataloader(),
        output_root / "train" / job_id,
        cfg.preprocess.train_num_writers,
        cfg.preprocess.shard_maxcount,
        cfg.preprocess.queue_size_per_worker,
    )


if __name__ == "__main__":
    main()
