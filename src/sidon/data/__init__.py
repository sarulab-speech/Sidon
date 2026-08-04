"""Dataset helpers for Sidon."""

from typing import Any

from .datamodule import PreprocessedDataModule, torch_audio

__all__ = ["PreprocessedDataModule", "WebDatasetDataModule", "torch_audio"]


def __getattr__(name: str) -> Any:
    if name == "WebDatasetDataModule":
        from .preprocess import WebDatasetDataModule

        return WebDatasetDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
