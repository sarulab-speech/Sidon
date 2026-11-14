"""Latent statistics of the SSL-VAE: per-dimension mean and variance of the posterior draws the separation model
is trained to predict.

The encoder sees what separation training feeds it: both clean speakers of dialogue segments
(statistics.segment_seconds), the training front end and attention, and the training precision. Draws come from the
global Torch stream after seeding with `seed`. The output records the encoder's weight digest, which
SSLVAE.load_statistics checks, and the posterior spread (informative only).

  python -m sidon.dialoguesidon2.statistics statistics.vae_checkpoint=<VAE-stage ckpt> \
      'statistics.train_urls=[<dialogue shard dir>]' statistics.output=<file>
"""

from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig

from sidon.model.dialoguesidon2.functional import length_mask

# Lightning precision -> autocast dtype of the training step (None: no autocast).
AUTOCAST = {"bf16-mixed": torch.bfloat16, "16-mixed": torch.float16, "32-true": None, "32": None}


class RunningStatistics:
    """Streaming mean and (population) variance in float64 over [N, D] rows."""

    def __init__(self, dim: int):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double()
        count = values.size(0)
        if not count:
            return
        mean = values.mean(0).cpu()
        m2 = (values - mean.to(values.device)).square().sum(0).cpu()
        delta = mean - self.mean
        total = self.count + count
        self.m2 += m2 + delta.square() * (self.count * count / total)
        self.mean += delta * (count / total)
        self.count = total

    def result(self) -> dict:
        if self.count < 2:
            raise ValueError("At least two valid frames are required")
        return {"mean": self.mean.float(), "var": (self.m2 / self.count).float(), "count": self.count}


class PosteriorStatistics:
    """Posterior spread per dimension: the mean over valid frames of exp(logvar / 2) against the standard
    deviation of mu over the same frames."""

    def __init__(self, dim: int):
        self.mu = RunningStatistics(dim)
        self.std_sum = torch.zeros(dim, dtype=torch.float64)

    def update(self, mu: torch.Tensor, logvar: torch.Tensor, lengths: torch.Tensor) -> None:
        valid = length_mask(lengths, mu.size(1))
        self.mu.update(mu[valid])
        self.std_sum += torch.exp(0.5 * logvar[valid].double()).sum(0).cpu()

    def view(self, vae):
        """A raw-latent encoder that records each posterior and returns the latent vae.raw() would, from the
        same w2v-BERT pass."""

        def raw(inputs):
            mu, logvar, lengths = vae.posterior(inputs)
            self.update(mu, logvar, lengths)
            return vae.draw(mu, logvar, lengths), lengths

        return SimpleNamespace(raw=raw)

    def result(self) -> dict:
        mu = self.mu.result()
        posterior_std = (self.std_sum / mu["count"]).float()
        mu_std = mu["var"].sqrt()

        def spread(value):
            return {"median": float(value.quantile(0.5)), "p90": float(value.quantile(0.9))}

        return {"posterior_std": posterior_std, "mu_std": mu_std, "mu_mean": mu["mean"], "count": mu["count"],
                "summary": {"posterior_std": spread(posterior_std), "mu_std": spread(mu_std)}}


def training_autocast(precision, device: str):
    precision = str(precision)
    if precision not in AUTOCAST:
        raise ValueError(f"Unsupported precision {precision} for statistics")
    dtype = AUTOCAST[precision]
    return torch.autocast(torch.device(device).type, dtype=dtype, enabled=dtype is not None)


def compute_statistics(cfg: DictConfig) -> dict:
    """Statistics over at most statistics.max_batches batches of the training dialogues (cfg.statistics)."""
    from sidon.dialoguesidon2.data import DialogueDataModule
    from sidon.model.dialoguesidon2.lightning_module import encode_speakers
    from sidon.model.dialoguesidon2.ssl_attention import optimize_relative_key_attention
    from sidon.model.dialoguesidon2.vae import SSLVAE, load_vae_checkpoint

    options = cfg.statistics
    if options.max_batches < 1:
        raise ValueError("statistics.max_batches must be positive")
    device = options.device
    precision = str(options.precision)
    training_autocast(precision, device)  # unsupported precisions fail before any work
    data = DialogueDataModule(list(options.train_urls), [], batch_size=int(options.batch_size),
                              num_workers=int(options.get("num_workers", 0)),
                              segment_seconds=options.segment_seconds)
    vae_cfg, vae_state = load_vae_checkpoint(options.vae_checkpoint)
    # Encoder only: the decoder is not needed here.
    vae = SSLVAE(vae_cfg, initialize_pretrained=False, decoder=False)
    vae.load_weights({k: v for k, v in vae_state.items() if not k.startswith("decoder.")})
    vae.sample_posterior = bool(options.get("sample_posterior", True))
    identity = vae.identity()
    efficient = bool(options.get("efficient_ssl_attention", True))
    if efficient:
        optimize_relative_key_attention(vae)
    vae = vae.to(device).eval()
    stats = RunningStatistics(vae.dim)
    posterior = PosteriorStatistics(vae.dim)
    view = posterior.view(vae)
    loader = data.loader(data.train_urls, training=False, repeat=False, segment_seconds=data.segment_seconds)
    batches = 0
    for batch in loader:
        with training_autocast(precision, device):
            z, lengths = encode_speakers(view.raw, batch, vae.mean.device)
        # [B, 2, T, D] latents; both speakers share the frame counts.
        stats.update(z[length_mask(lengths, z.size(2))[:, None].expand(-1, 2, -1)])
        batches += 1
        if batches % 100 == 0:
            print(f"Statistics: {batches} batches, {stats.count} frames", flush=True)
        if batches >= options.max_batches:
            break
    return {**stats.result(), "latent_dim": vae.dim, "sample_posterior": vae.sample_posterior,
            "encoder_identity": identity, "segment_seconds": data.segment_seconds, "precision": precision,
            "efficient_ssl_attention": efficient, "seed": int(cfg.seed), "batches": batches,
            "training_shards": len(data.train_urls), "posterior": posterior.result()}


@hydra.main(config_path="../../../config", config_name="dialoguesidon2", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    output = Path(cfg.statistics.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result = compute_statistics(cfg)
    torch.save(result, temporary)
    temporary.replace(output)
    summary = result["posterior"]["summary"]
    print(f"Saved statistics from {result['batches']} batches ({result['count']} frames) to {output}; "
          f"posterior std median {summary['posterior_std']['median']:.4g}, per-dimension std of mu median "
          f"{summary['mu_std']['median']:.4g}")


if __name__ == "__main__":
    main()
