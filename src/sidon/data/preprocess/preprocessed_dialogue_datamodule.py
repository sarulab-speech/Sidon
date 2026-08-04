from __future__ import annotations

import io
import importlib
import json
import re
import tarfile
from typing import Any, Iterable, Iterator, NoReturn, Sequence

import torch
import torch.nn.functional as F

from ..datamodule import PreprocessedDataModule, skip_nan_preprocessed


wds = importlib.import_module("webdataset")
wds_filters = importlib.import_module("webdataset.filters")

_MAX_TENSOR_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_METADATA_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_EXTENSION_HEADER_BYTES = 4 * 1024
_ALLOWED_TENSOR_FIELDS = frozenset(
    {
        ".input_wav.pth",
        ".noisy_input_wav.pth",
        ".noisy_input_wav16k.pth",
        ".clean_mixture.pth",
        ".clean_16k_mixture.pth",
        ".noisy_mixture.pth",
        ".noisy_16k_mixture.pth",
    }
)
_OPTIONAL_TENSOR_FIELDS = (
    "noisy_input_wav.pth",
    "noisy_input_wav16k.pth",
    "clean_mixture.pth",
    "clean_16k_mixture.pth",
    "noisy_mixture.pth",
    "noisy_16k_mixture.pth",
)
_MEMBER_SIZE_LIMITS = {
    **{
        field.removeprefix("."): _MAX_TENSOR_PAYLOAD_BYTES
        for field in _ALLOWED_TENSOR_FIELDS
    },
    "sr.index": 64,
    "manifest.json": _MAX_METADATA_PAYLOAD_BYTES,
}
_RECORD_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,254}")


class BoundedTarInfo(tarfile.TarInfo):
    def _validate_extension_header_size(self) -> None:
        if self.size < 0 or self.size > _MAX_EXTENSION_HEADER_BYTES:
            raise ValueError("tar extension header exceeds size limit")

    def _proc_gnulong(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        self._validate_extension_header_size()
        return getattr(super(), "_proc_gnulong")(archive)

    def _proc_pax(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        self._validate_extension_header_size()
        return getattr(super(), "_proc_pax")(archive)

    def _proc_sparse(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        del archive
        self._reject_sparse_entry()

    def _proc_gnusparse_00(self, next_member: Any, raw_headers: Any) -> NoReturn:
        del next_member, raw_headers
        self._reject_sparse_entry()

    def _proc_gnusparse_01(self, next_member: Any, pax_headers: Any) -> NoReturn:
        del next_member, pax_headers
        self._reject_sparse_entry()

    def _proc_gnusparse_10(
        self,
        next_member: Any,
        pax_headers: Any,
        archive: tarfile.TarFile,
    ) -> NoReturn:
        del next_member, pax_headers, archive
        self._reject_sparse_entry()

    @staticmethod
    def _reject_sparse_entry() -> NoReturn:
        raise ValueError("tar sparse entries are disabled")


def _parse_member_name(name: str) -> tuple[str, str]:
    for suffix in sorted(_MEMBER_SIZE_LIMITS, key=len, reverse=True):
        separator = f".{suffix}"
        if name.endswith(separator):
            record_key = name[: -len(separator)]
            if _RECORD_KEY_PATTERN.fullmatch(record_key) is None:
                raise ValueError(f"tar member has an unsafe record key: {name!r}")
            return record_key, suffix
    raise ValueError(f"tar member has an unsupported suffix: {name!r}")


def bounded_tar_file_iterator(fileobj: Any) -> Iterator[dict[str, Any]]:
    with tarfile.open(
        fileobj=fileobj,
        mode="r|*",
        tarinfo=BoundedTarInfo,
    ) as archive:
        for tar_info in archive:
            if not tar_info.isreg():
                raise ValueError(f"tar member must be a regular file: {tar_info.name!r}")
            _, suffix = _parse_member_name(tar_info.name)
            size_limit = _MEMBER_SIZE_LIMITS[suffix]
            if tar_info.size < 0 or tar_info.size > size_limit:
                raise ValueError(
                    f"tar member exceeds size limit: {tar_info.name!r}, "
                    f"size={tar_info.size}, limit={size_limit}"
                )
            extracted = archive.extractfile(tar_info)
            if extracted is None:
                raise ValueError(f"cannot read tar member: {tar_info.name!r}")
            yield {"fname": tar_info.name, "data": extracted.read()}
            setattr(archive, "members", [])


def bounded_tar_file_expander(
    sources: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for source in sources:
        if "stream" not in source or "url" not in source:
            raise ValueError("WebDataset source must contain stream and url")
        for sample in bounded_tar_file_iterator(source["stream"]):
            sample["__url__"] = source["url"]
            if "local_path" in source:
                sample["__local_path__"] = source["local_path"]
            yield sample
        yield {}


_bounded_tar_file_expander_stage = wds_filters.pipelinefilter(
    bounded_tar_file_expander
)


def _bounded_webdataset(urls: Sequence[str], *, shardshuffle: int | bool) -> Any:
    dataset = wds.WebDataset(
        urls,
        shardshuffle=shardshuffle,
        nodesplitter=lambda values: values,
        workersplitter=wds.split_by_worker,
        repeat=True,
        empty_check=True,
        handler=wds.reraise_exception,
    )
    expander_indices = [
        index
        for index, pipeline_stage in enumerate(dataset.pipeline)
        if getattr(getattr(pipeline_stage, "f", None), "__name__", None)
        == "tar_file_expander"
    ]
    if len(expander_indices) != 1:
        raise RuntimeError("unsupported WebDataset pipeline: tar expander not found")
    dataset.pipeline[expander_indices[0]] = _bounded_tar_file_expander_stage()
    return dataset


def safe_dialogue_decoder(key: str, data: bytes) -> object | None:
    normalized = key.lower()
    if normalized.endswith((".pickle", ".pkl")):
        raise ValueError(f"pickle payloads are disabled: {key}")
    if normalized.endswith(".pth"):
        if normalized not in _ALLOWED_TENSOR_FIELDS:
            raise ValueError(f"tensor payload is not allowlisted: {key}")
        if not isinstance(data, bytes):
            raise TypeError(f"{key} must contain bytes")
        if len(data) > _MAX_TENSOR_PAYLOAD_BYTES:
            raise ValueError(f"{key} exceeds the tensor payload size limit")
        try:
            value = torch.load(
                io.BytesIO(data),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise ValueError(f"{key} is not a safe tensor payload") from error
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must contain exactly one tensor")
        if value.numel() * value.element_size() > _MAX_TENSOR_PAYLOAD_BYTES:
            raise ValueError(f"{key} decoded tensor exceeds the size limit")
        return value
    if normalized.endswith(".index"):
        if not isinstance(data, bytes) or len(data) > 64:
            raise ValueError(f"{key} is not a valid index payload")
        try:
            return int(data.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError(f"{key} is not a valid integer index") from error
    if normalized.endswith(".json"):
        if not isinstance(data, bytes) or len(data) > _MAX_METADATA_PAYLOAD_BYTES:
            raise ValueError(f"{key} exceeds the JSON payload size limit")
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{key} is not valid JSON") from error
    return None


class PreprocessedDialogueDataModule(PreprocessedDataModule):
    """Load channel-preserving dialogue tensors from completed shards."""

    def __init__(
        self,
        train_urls: Sequence[str] | str,
        val_urls: Sequence[str] | str,
        batch_size: int,
        val_batch_size: int,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
        preprocessed: bool = True,
        is_s3: bool = False,
        required_sample_rate: int | None = None,
        expected_duration_seconds: int | None = None,
        required_input_channels: int | None = None,
        require_noisy_16k_mixture: bool = False,
    ) -> None:
        super().__init__(
            train_urls=train_urls,
            val_urls=val_urls,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            preprocessed=preprocessed,
            is_s3=is_s3,
        )
        if required_sample_rate is not None and (
            isinstance(required_sample_rate, bool)
            or not isinstance(required_sample_rate, int)
            or required_sample_rate <= 0
        ):
            raise ValueError("required_sample_rate must be a positive integer")
        if expected_duration_seconds is not None and (
            isinstance(expected_duration_seconds, bool)
            or not isinstance(expected_duration_seconds, int)
            or expected_duration_seconds <= 0
        ):
            raise ValueError("expected_duration_seconds must be a positive integer")
        if required_input_channels is not None and (
            isinstance(required_input_channels, bool)
            or not isinstance(required_input_channels, int)
            or required_input_channels <= 0
        ):
            raise ValueError("required_input_channels must be a positive integer")
        if not isinstance(require_noisy_16k_mixture, bool):
            raise ValueError("require_noisy_16k_mixture must be boolean")
        self.required_sample_rate = required_sample_rate
        self.expected_duration_seconds = expected_duration_seconds
        self.required_input_channels = required_input_channels
        self.require_noisy_16k_mixture = require_noisy_16k_mixture

    def setup(self, stage: str | None = None) -> None:
        del stage
        self.train_dataset = (
            _bounded_webdataset(self.train_urls, shardshuffle=100)
            .decode(
                safe_dialogue_decoder,
                handler=wds.reraise_exception,
                pre=[],
                post=[],
            )
            .compose(skip_nan_preprocessed)
            .shuffle(100)
            .batched(self.batch_size, collation_fn=self.collate_fn)
        )
        self.val_dataset = (
            _bounded_webdataset(self.val_urls, shardshuffle=False)
            .decode(
                safe_dialogue_decoder,
                handler=wds.reraise_exception,
                pre=[],
                post=[],
            )
            .compose(skip_nan_preprocessed)
            .batched(self.val_batch_size, collation_fn=self.collate_fn)
        )

    @staticmethod
    def _as_batched(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        return tensor

    @classmethod
    def _pad_cat(cls, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.empty(0)
        batched = [cls._as_batched(tensor) for tensor in tensors]
        max_length = max(tensor.size(-1) for tensor in batched)
        padded = [
            F.pad(tensor, (0, max_length - tensor.size(-1)))
            for tensor in batched
        ]
        return torch.cat(padded, dim=0)

    def _strict_contract_enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.required_sample_rate,
                self.expected_duration_seconds,
                self.required_input_channels,
            )
        ) or self.require_noisy_16k_mixture

    def _validate_samples(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            raise ValueError("preprocessed dialogue batch must not be empty")
        strict = self._strict_contract_enabled()
        required_keys = {"input_wav.pth"}
        if strict:
            required_keys.add("sr.index")
        if self.require_noisy_16k_mixture:
            required_keys.add("noisy_16k_mixture.pth")
        for index, sample in enumerate(samples):
            missing = sorted(required_keys - sample.keys())
            if missing:
                raise ValueError(
                    f"sample {index} is missing required payloads: {', '.join(missing)}"
                )
            input_wav = sample["input_wav.pth"]
            if not isinstance(input_wav, torch.Tensor) or not input_wav.is_floating_point():
                raise ValueError(f"sample {index} input_wav.pth must be a float tensor")
            if input_wav.dim() not in (1, 2):
                raise ValueError(
                    f"sample {index} input_wav.pth must not contain an inner batch dimension"
                )
            if not torch.isfinite(input_wav).all():
                raise ValueError(f"sample {index} input_wav.pth must be finite")
            for payload_key in _OPTIONAL_TENSOR_FIELDS:
                if payload_key not in sample:
                    continue
                payload = sample[payload_key]
                if not isinstance(payload, torch.Tensor) or not payload.is_floating_point():
                    raise ValueError(
                        f"sample {index} {payload_key} must be a float tensor"
                    )
                if payload.dim() not in (1, 2):
                    raise ValueError(
                        f"sample {index} {payload_key} must not contain an inner batch dimension"
                    )
                if not torch.isfinite(payload).all():
                    raise ValueError(f"sample {index} {payload_key} must be finite")
            if not strict:
                continue
            if self.required_input_channels is not None and (
                input_wav.dim() != 2
                or input_wav.size(0) != self.required_input_channels
            ):
                raise ValueError(
                    f"sample {index} input_wav.pth must have "
                    f"{self.required_input_channels} channels"
                )
            if self.required_sample_rate is not None and (
                sample["sr.index"] != self.required_sample_rate
            ):
                raise ValueError(
                    f"sample {index} sr.index must be {self.required_sample_rate}"
                )
            if (
                self.expected_duration_seconds is not None
                and self.required_sample_rate is not None
                and input_wav.size(-1)
                != self.expected_duration_seconds * self.required_sample_rate
            ):
                raise ValueError(
                    f"sample {index} input_wav.pth has an unexpected duration"
                )
            if self.require_noisy_16k_mixture:
                mixture = sample["noisy_16k_mixture.pth"]
                if (
                    not isinstance(mixture, torch.Tensor)
                    or not mixture.is_floating_point()
                    or mixture.dim() != 1
                ):
                    raise ValueError(
                        f"sample {index} noisy_16k_mixture.pth must be a 1D float tensor"
                    )
                if not torch.isfinite(mixture).all():
                    raise ValueError(
                        f"sample {index} noisy_16k_mixture.pth must be finite"
                    )
                if (
                    self.expected_duration_seconds is not None
                    and mixture.size(-1) != self.expected_duration_seconds * 16_000
                ):
                    raise ValueError(
                        f"sample {index} noisy_16k_mixture.pth has an unexpected duration"
                    )

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        self._validate_samples(samples)
        input_wavs = [sample["input_wav.pth"] for sample in samples]
        for key in _OPTIONAL_TENSOR_FIELDS:
            present = sum(key in sample for sample in samples)
            if present not in (0, len(samples)):
                raise ValueError(f"{key} must be present for every sample or none")
        optional = {
            key: [sample[key] for sample in samples if key in sample]
            for key in _OPTIONAL_TENSOR_FIELDS
        }
        input_wav = self._pad_cat(input_wavs)
        input_wav_lens = torch.tensor(
            [int(self._as_batched(tensor).size(-1)) for tensor in input_wavs],
            dtype=torch.long,
        )
        names: list[str] = []
        for sample in samples:
            if "names.pickle" in sample:
                name = sample["names.pickle"]
                if isinstance(name, list):
                    names.extend(str(value) for value in name)
                else:
                    names.append(str(name))
            elif "__key__" in sample:
                names.append(str(sample["__key__"]))
        batch: dict[str, Any] = {
            "input_wav": input_wav.float(),
            "input_wav_lens": input_wav_lens,
            "sr": samples[0].get("sr.index", 48_000),
            "names": names,
        }
        output_names = {
            "noisy_input_wav.pth": ("noisy_input_wav", False),
            "noisy_input_wav16k.pth": ("noisy_input_wav16k", False),
            "clean_mixture.pth": ("clean_mixture", True),
            "clean_16k_mixture.pth": ("clean_16k_mixture", True),
            "noisy_mixture.pth": ("noisy_mixture", True),
            "noisy_16k_mixture.pth": ("noisy_16k_mixture", True),
        }
        for payload_key, tensors in optional.items():
            if not tensors:
                continue
            output_key, squeeze_channel = output_names[payload_key]
            value = self._pad_cat(tensors)
            batch[output_key] = value.squeeze(1).float() if squeeze_channel else value.float()
        return batch
