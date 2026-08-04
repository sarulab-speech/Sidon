import io
import importlib
import gzip
import os
from pathlib import Path
import shlex
import sys
import tarfile

import pytest
import torch


pytestmark = [
    pytest.mark.filterwarnings("ignore:builtin type Swig.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:builtin type swigvarlink.*:DeprecationWarning"),
]


def preprocessed_datamodule_class():
    module = importlib.import_module(
        "sidon.data.preprocess.preprocessed_dialogue_datamodule"
    )
    return module.PreprocessedDialogueDataModule


def add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


class MaliciousPayload:
    def __init__(self, command: str) -> None:
        self.command = command

    def __reduce__(self):
        return os.system, (self.command,)


def write_fixture_shard(root: Path) -> None:
    shard = root / "fixture.tar"
    with tarfile.open(shard, "w") as archive:
        for index in range(2):
            key = f"fixture-{index}"
            clean = torch.zeros((2, 480_000), dtype=torch.float32)
            clean[index, :100] = 0.25
            mixture = torch.zeros(320_000, dtype=torch.float32)
            mixture[:100] = 0.125
            add_member(archive, f"{key}.input_wav.pth", tensor_bytes(clean))
            add_member(
                archive,
                f"{key}.noisy_16k_mixture.pth",
                tensor_bytes(mixture),
            )
            add_member(archive, f"{key}.sr.index", b"24000")


def strict_datamodule(root: Path, *, batch_size: int = 1):
    PreprocessedDialogueDataModule = preprocessed_datamodule_class()
    return PreprocessedDialogueDataModule(
        train_urls=[str(root)],
        val_urls=[str(root)],
        batch_size=batch_size,
        val_batch_size=batch_size,
        required_sample_rate=24_000,
        expected_duration_seconds=20,
        required_input_channels=2,
        require_noisy_16k_mixture=True,
    )


def test_preprocessed_loader_does_not_import_raw_augmentation_dependencies(
    tmp_path: Path,
) -> None:
    PreprocessedDialogueDataModule = preprocessed_datamodule_class()

    assert "sidon.data.preprocess.functional_degrations" not in sys.modules
    write_fixture_shard(tmp_path)
    datamodule = PreprocessedDialogueDataModule(
        train_urls=[str(tmp_path)],
        val_urls=[str(tmp_path)],
        batch_size=2,
        val_batch_size=2,
        train_num_workers=0,
        val_num_workers=0,
        preprocessed=True,
        is_s3=False,
        required_sample_rate=24_000,
        expected_duration_seconds=20,
        required_input_channels=2,
        require_noisy_16k_mixture=True,
    )

    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataset))

    assert batch["input_wav"].shape == (2, 2, 480_000)
    assert batch["input_wav_lens"].tolist() == [480_000, 480_000]
    assert batch["noisy_16k_mixture"].shape == (2, 320_000)
    assert batch["sr"] == 24_000


def test_legacy_dialogue_configs_keep_legacy_loader() -> None:
    root = Path(__file__).resolve().parents[1]
    target = (
        "_target_: "
        "sidon.data.preprocess.dialogue_datamodule.PreprocessedDialogueDataModule"
    )

    for config_name in ("dialogue_preprocessed.yaml", "dialogue_preprocessed_120s.yaml"):
        config = root / "config" / "data" / config_name
        assert target in config.read_text(encoding="utf-8")


def test_lazy_webdataset_exports_preserve_public_api() -> None:
    data_package = importlib.import_module("sidon.data")
    preprocess_package = importlib.import_module("sidon.data.preprocess")

    assert "WebDatasetDataModule" in data_package.__all__
    assert "WebDatasetDataModule" in preprocess_package.__all__


def test_strict_contract_rejects_non_finite_or_missing_payloads(
    tmp_path: Path,
) -> None:
    PreprocessedDialogueDataModule = preprocessed_datamodule_class()

    datamodule = PreprocessedDialogueDataModule(
        train_urls=[str(tmp_path)],
        val_urls=[str(tmp_path)],
        batch_size=1,
        val_batch_size=1,
        required_sample_rate=24_000,
        expected_duration_seconds=20,
        required_input_channels=2,
        require_noisy_16k_mixture=True,
    )
    clean = torch.zeros((2, 480_000), dtype=torch.float32)
    mixture = torch.zeros(320_000, dtype=torch.float32)
    mixture[0] = torch.inf

    with pytest.raises(ValueError, match="finite"):
        datamodule.collate_fn(
            [
                {
                    "input_wav.pth": clean,
                    "noisy_16k_mixture.pth": mixture,
                    "sr.index": 24_000,
                }
            ]
        )
    with pytest.raises(ValueError, match="noisy_16k_mixture.pth"):
        datamodule.collate_fn(
            [{"input_wav.pth": clean, "sr.index": 24_000}]
        )


def test_collate_rejects_optional_payload_present_for_only_part_of_batch(
    tmp_path: Path,
) -> None:
    PreprocessedDialogueDataModule = preprocessed_datamodule_class()
    datamodule = PreprocessedDialogueDataModule(
        train_urls=[str(tmp_path)],
        val_urls=[str(tmp_path)],
        batch_size=2,
        val_batch_size=2,
    )
    clean = torch.zeros((2, 480_000), dtype=torch.float32)

    with pytest.raises(ValueError, match="noisy_mixture.pth.*every sample"):
        datamodule.collate_fn(
            [
                {
                    "input_wav.pth": clean,
                    "noisy_mixture.pth": torch.zeros(480_000),
                    "sr.index": 24_000,
                },
                {"input_wav.pth": clean, "sr.index": 24_000},
            ]
        )


@pytest.mark.parametrize(
    ("payload_key", "payload"),
    (
        ("input_wav.pth", torch.zeros((1, 2, 8))),
        ("noisy_mixture.pth", torch.zeros((1, 1, 8))),
    ),
)
def test_collate_rejects_hidden_inner_batch_dimension(
    tmp_path: Path,
    payload_key: str,
    payload: torch.Tensor,
) -> None:
    PreprocessedDialogueDataModule = preprocessed_datamodule_class()
    datamodule = PreprocessedDialogueDataModule(
        train_urls=[str(tmp_path)],
        val_urls=[str(tmp_path)],
        batch_size=1,
        val_batch_size=1,
    )
    sample = {
        "input_wav.pth": torch.zeros((2, 8)),
        "sr.index": 24_000,
        payload_key: payload,
    }

    with pytest.raises(ValueError, match="must not contain an inner batch dimension"):
        datamodule.collate_fn([sample])


def test_get_urls_shell_quotes_valid_s3_uris(tmp_path: Path) -> None:
    get_urls = importlib.import_module("sidon.data.datamodule").get_urls
    uris = (
        "s3://bucket/dialogue shard.tar",
        "s3://bucket/dialogue;touch /tmp/not-executed.tar",
    )
    manifest = tmp_path / "s3.txt"
    manifest.write_text("\n".join(uris), encoding="utf-8")

    urls = get_urls(str(manifest))

    assert len(urls) == len(uris)
    for url, uri in zip(urls, uris):
        assert shlex.split(url.removeprefix("pipe:")) == [
            "aws",
            "--endpoint-url",
            "https://s3ds.mdx.jp",
            "s3",
            "cp",
            uri,
            "-",
        ]


@pytest.mark.parametrize(
    "uri",
    ("https://bucket/shard.tar", "s3:///shard.tar", "s3://bucket"),
)
def test_get_urls_rejects_invalid_s3_uris(tmp_path: Path, uri: str) -> None:
    get_urls = importlib.import_module("sidon.data.datamodule").get_urls
    manifest = tmp_path / "s3.txt"
    manifest.write_text(uri, encoding="utf-8")

    with pytest.raises(ValueError, match="S3 URI"):
        get_urls(str(manifest))


def test_safe_dialogue_decoder_loads_allowlisted_tensor() -> None:
    decoder = importlib.import_module(
        "sidon.data.preprocess.preprocessed_dialogue_datamodule"
    ).safe_dialogue_decoder
    expected = torch.arange(4, dtype=torch.float32)

    actual = decoder(".input_wav.pth", tensor_bytes(expected))

    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual, expected)


def test_safe_dialogue_decoder_rejects_unknown_tensor_member() -> None:
    decoder = importlib.import_module(
        "sidon.data.preprocess.preprocessed_dialogue_datamodule"
    ).safe_dialogue_decoder

    with pytest.raises(ValueError, match="not allowlisted"):
        decoder(".unexpected.pth", tensor_bytes(torch.zeros(1)))


def test_safe_dialogue_decoder_rejects_pickle_code_execution(
    tmp_path: Path,
) -> None:
    decoder = importlib.import_module(
        "sidon.data.preprocess.preprocessed_dialogue_datamodule"
    ).safe_dialogue_decoder
    marker = tmp_path / "executed"
    buffer = io.BytesIO()
    torch.save(MaliciousPayload(f"touch {shlex.quote(str(marker))}"), buffer)

    with pytest.raises(ValueError, match="safe tensor"):
        decoder(".input_wav.pth", buffer.getvalue())

    assert not marker.exists()


def test_safe_dialogue_decoder_rejects_pickle_members() -> None:
    decoder = importlib.import_module(
        "sidon.data.preprocess.preprocessed_dialogue_datamodule"
    ).safe_dialogue_decoder

    with pytest.raises(ValueError, match="pickle payloads are disabled"):
        decoder(".names.pickle", b"not loaded")


def test_loader_rejects_unsafe_record_key_before_decoding(tmp_path: Path) -> None:
    shard = tmp_path / "unsafe-key.tar"
    with tarfile.open(shard, "w") as archive:
        add_member(
            archive,
            "../escape.input_wav.pth",
            tensor_bytes(torch.zeros((2, 480_000), dtype=torch.float32)),
        )
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="unsafe record key"):
        next(iter(datamodule.train_dataset))


def test_loader_rejects_dotted_record_key_before_grouping(tmp_path: Path) -> None:
    shard = tmp_path / "dotted-key.tar"
    with tarfile.open(shard, "w") as archive:
        add_member(
            archive,
            "speaker.001.input_wav.pth",
            tensor_bytes(torch.zeros((2, 480_000), dtype=torch.float32)),
        )
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="unsafe record key"):
        next(iter(datamodule.train_dataset))


def test_loader_rejects_declared_oversized_member_before_reading(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "oversized.tar"
    info = tarfile.TarInfo("fixture.input_wav.pth")
    info.size = 64 * 1024 * 1024 + 1
    shard.write_bytes(info.tobuf(format=tarfile.GNU_FORMAT) + (b"\0" * 1024))
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="exceeds size limit"):
        next(iter(datamodule.train_dataset))


@pytest.mark.parametrize(
    "type_flag",
    (tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE),
)
def test_loader_rejects_extended_header_before_payload_read(
    tmp_path: Path,
    type_flag: bytes,
) -> None:
    shard = tmp_path / "extended-header.tar"
    info = tarfile.TarInfo("././@LongLink")
    info.type = type_flag
    info.size = 64 * 1024 * 1024 + 1
    shard.write_bytes(info.tobuf(format=tarfile.GNU_FORMAT) + (b"\0" * 1024))
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="extension header exceeds size limit"):
        next(iter(datamodule.train_dataset))


def test_loader_rejects_sparse_header_before_payload_read(tmp_path: Path) -> None:
    shard = tmp_path / "sparse-header.tar"
    info = tarfile.TarInfo("fixture.input_wav.pth")
    info.type = tarfile.GNUTYPE_SPARSE
    info.size = 1
    shard.write_bytes(info.tobuf(format=tarfile.GNU_FORMAT) + (b"\0" * 1024))
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="sparse entries are disabled"):
        next(iter(datamodule.train_dataset))


@pytest.mark.parametrize(
    "pax_headers",
    (
        {
            "GNU.sparse.size": "1",
            "GNU.sparse.offset": "0",
            "GNU.sparse.numbytes": "1",
        },
        {"GNU.sparse.map": "0,1"},
        {"GNU.sparse.major": "1", "GNU.sparse.minor": "0"},
    ),
)
def test_loader_rejects_pax_sparse_variants(
    tmp_path: Path,
    pax_headers: dict[str, str],
) -> None:
    shard = tmp_path / "pax-sparse.tar"
    info = tarfile.TarInfo("fixture.input_wav.pth")
    info.size = 1
    info.pax_headers = pax_headers
    with tarfile.open(shard, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(info, io.BytesIO(b"\0"))
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="sparse entries are disabled"):
        next(iter(datamodule.train_dataset))


def test_loader_rejects_non_regular_member(tmp_path: Path) -> None:
    shard = tmp_path / "link-member.tar"
    info = tarfile.TarInfo("fixture.input_wav.pth")
    info.type = tarfile.SYMTYPE
    info.linkname = "other.input_wav.pth"
    with tarfile.open(shard, "w") as archive:
        archive.addfile(info)
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="regular file"):
        next(iter(datamodule.train_dataset))


def test_loader_rejects_gzip_member_without_decompressing(tmp_path: Path) -> None:
    shard = tmp_path / "gzip-member.tar"
    compressed = gzip.compress(
        tensor_bytes(torch.zeros((2, 480_000), dtype=torch.float32))
    )
    with tarfile.open(shard, "w") as archive:
        add_member(archive, "fixture.input_wav.pth.gz", compressed)
    datamodule = strict_datamodule(tmp_path)
    datamodule.setup("fit")

    with pytest.raises(ValueError, match="unsupported suffix"):
        next(iter(datamodule.train_dataset))
