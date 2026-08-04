"""Preprocessing data modules and utilities for creating WebDataset shards."""

from typing import Any

__all__ = ["WebDatasetDataModule"]


def __getattr__(name: str) -> Any:
    if name == "WebDatasetDataModule":
        from .webdataset_datamodule import WebDatasetDataModule

        return WebDatasetDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
