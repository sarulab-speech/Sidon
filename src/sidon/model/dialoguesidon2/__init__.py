"""DialogueSidon2: two-speaker dialogue separation and restoration.

A fine-tuned w2v-BERT 2.0 encoder reads the noisy mixture and a small readout head predicts, for each of the two
speakers, the latent of an SSL-VAE (w2v-BERT 2.0 features compressed by a VAE bottleneck, decoded to 24 kHz audio
by a DAC decoder). Training has two stages: the SSL-VAE (``SSLVAEModule``) and the separation model
(``DialogueSidon2``), which learns to predict the VAE's latents of the clean speakers with a permutation-invariant
loss.
"""
