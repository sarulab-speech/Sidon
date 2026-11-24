# DialogueSidon2 training data

This directory holds the code that produced the training data of DialogueSidon2. The model is trained on
two-speaker conversations stored as WebDataset shards. Each record holds two clean per-speaker target channels,
their degraded versions and the two-channel mixtures. The data comes from four roots, which share one shard format
and one producer:

| Root | Speech | Restoration | Selection | Source keys |
| --- | --- | --- | --- | --- |
| base | CallHome (German, English, Japanese, Spanish, Mandarin) and Fisher English, stereo telephone conversations | Sidon v0.1 | none (whole conversations) | uuid4 |
| OpenYap-1k | per-speaker tracks of two-party conversations | Sidon v0.1 | metadata screen, crosstalk screen, post-restoration check | `openyap-<12 hex>-<offset s>` |
| Basis | per-speaker tracks of 2-28-party conversations, one pair per window | Sidon v0.1 | metadata screen, checksums, crosstalk screen, post-restoration check | `basis-<conversation>-<segment start>-<offset s>` |
| SIM | simulated two-speaker conversations from LibriTTS-R utterances | none (clean speech) | consistency checks of the simulation | `sim-ltr-<12 hex>-<n>` |

Every root ends in **source tars**: stereo 48 kHz FLAC records, one speaker per channel. For OpenYap, Basis and SIM
each record is one 120 s window; for the base root each record is a whole conversation. The **producer** cuts the
records into 120 s windows, degrades each channel and writes the training shards.

```
scripts/dialoguesidon2/data/
  common/     shared helpers: crosstalk statistics, decisions, layout and tars, restoration, shard checks
  openyap/    OpenYap-1k pipeline           run/openyap.sh
  basis/      Basis pipeline                run/basis.sh
  sim/        simulated conversations       run/sim.sh
  producer/   the producer                  run/produce.sh, config/data/dialoguesidon2_producer.yaml
  run/        stage scripts (env.sh is sourced by all; restore.sh runs the restoration on GPUs)
```

Paths in this README are relative to `scripts/dialoguesidon2/data/`, except `config/` and `tests/` at the top of
the repository.

## Reproducibility

The runs that made the released data were **not seeded end to end**, so the outputs of this recipe are not
bit-identical to the released shards. In particular:

* The producer is not seeded. Python's `random` drives every degradation and the mixing ratio. The record keys are
  uuid4 values, and the shard order depends on shard shuffling and on worker scheduling.
* The base root's source tars use uuid4 keys.
* Restoration runs on GPUs, which need not be bit-exact across hardware and library versions.

The selection steps are deterministic, given the same inputs and thresholds:

* the screens and their decisions;
* the speaker-disjoint dev splits;
* the assignment of windows to source tars (seeded: OpenYap 20261012, Basis 20261013);
* the Basis pair choice and channel order (sha256 with seed 20261013);
* the SIM plan and rendering (seed 20261015).

Each stage writes a JSON record of its parameters, the code commit and its counts next to its outputs.

## Requirements

* The Sidon environment of this repository (`pyproject.toml`). The stage scripts import `sidon` from `src/`
  (`run/produce.sh` puts it on `PYTHONPATH`) and the helpers from this directory.
* pyroomacoustics with a C++ runtime that matches its compiled extension.
* torchaudio with FFmpeg (the mp3 degradation uses `torchaudio.io.AudioEffector`). The producer's codec step returns
  the sample unchanged when the encoder fails, so `producer/preflight.py` checks all five mp3 settings before a run.
* soundfile, scipy, numpy and webdataset; `onnxruntime` and `librosa` for the optional Basis DNSMOS step.
* GPUs for the restoration (`run/restore.sh`, one process per GPU). Everything else runs on CPUs.
* The Sidon v0.1 model is downloaded from the Hugging Face Hub (`sarulab-speech/sidon-v0.1`, revision
  `89624152d11da2d86b61dbb43d5b0ccc72b2cd9b`), together with the `facebook/w2v-bert-2.0` and
  `microsoft/wavlm-base-plus-sv` feature-extractor configurations that the producer's data module loads.
* Noise shards for the producer: WebDataset tars of noise recordings whose member names contain `audio`
  (for example `<id>.audio.wav`), any sample rate (resampled), mono or multichannel (the first channel is used).

The stage scripts in `run/` take their settings from environment variables. Every script sources `run/env.sh`,
which provides:

* `PYTHON` (default `python`);
* `PROCESSES` (default: the available CPUs, at most 64);
* `SCRATCH` (fast local scratch);
* `LOG_DIR` (stage logs).

The scripts submit no jobs; wrap each stage in your scheduler's job script. Every stage can be resumed: finished
units are skipped and partial outputs are redone.

## Restoration with Sidon v0.1

`common/sidon_restore.py` restores every channel of a file independently:

1. peak-normalise the channel to 0.9;
2. high-pass at 50 Hz;
3. resample to 16 kHz and append 0.5 s of silence;
4. cut into 96 s chunks;
5. per chunk, compute an 80-bin filterbank (w2v-BERT 2.0 layout) and run the TorchScript feature extractor and
   decoder; the last feature frame of each chunk is carried into the next;
6. trim the result to the input length at 48 kHz.

The output is a 48 kHz, 16-bit FLAC (round(x * 32768), clipped). Inputs `<input_dir>/**/*.flac` (`--pattern`) are
written to `<output_dir>/<input_dir name>/<relative path>.flac`.

```
bash run/restore.sh OUTPUT_DIR TREE [TREE ...]     # one process per GPU, each takes every n-th file
```

The corpus scripts build a tree of symlinks that holds only the tracks still without a valid output, call
`run/restore.sh`, and then check every expected output. An output is valid when it is mono 48 kHz PCM_16, has
exactly the raw length and its last 0.1 s decodes.

## The base root (CallHome and Fisher)

The base root uses CallHome (German, English, Japanese, Spanish, Mandarin) and Fisher English, which are LDC
corpora. You need your own LDC licences, and this recipe ships none of their data.

1. Convert each conversation to a stereo WAV or FLAC file, one speaker per channel (for example with `sph2pipe`).
   Keep one directory per corpus.
2. Restore the files of each corpus directory:
   `python common/sidon_restore.py --input_dir DIR --output_dir RESTORED --pattern '**/*.wav'` (one process per
   GPU, with `--num_shards/--shard_idx`).
3. Pack each restored corpus directory into source tars with uuid4 keys (records: `flac`, `original_path.txt`):
   ```
   python common/pack_sources.py --restored-root RESTORED --input-dir RESTORED/<corpus> --output-root SOURCE/train
   ```
   Put held-out stereo conversations, packed the same way, under `SOURCE/dev`. The validation shards are written
   once, by producer task 0, and are not used for training.
4. Run the producer: `NOISE_DIR=... bash run/produce.sh SOURCE OUT dialogue_24k_120s`. Without `manifest.tsv` in
   `SOURCE`, the record counts are not checked. The producer cuts each conversation into 120 s windows; the window
   that starts at sample o (48 kHz) of the restored conversation is named `<uuid4>_<o>`.

## OpenYap-1k

Source: the Hugging Face dataset `TheAgenticDataCompany/open-yap-1k`, CC BY 4.0.

`OY_ROOT` is the release's `conversations/` directory. Each `conv_<12 hex>/` subdirectory holds:

* `speaker_{a,b}.flac` (48 kHz, mono, 16-bit);
* `speaker_{a,b}_transcript.json` (word timings);
* `speaker_{a,b}_meta.json` (speaker id, headphones, audio metrics);
* `speaker_{a,b}_dnsmos.json` (a 3 s DNSMOS series);
* `meta.json`.

`OY_WORK` receives all outputs. Run the stages in this order:

```
export OY_ROOT=... OY_WORK=...
bash run/openyap.sh prescreen   # CPU, metadata only
bash run/openyap.sh scan        # CPU, transcripts only
bash run/openyap.sh screen      # CPU, decodes every raw track
bash run/openyap.sh restore     # GPU
bash run/openyap.sh cut         # CPU
NOISE_DIR=... bash run/openyap.sh produce
bash run/openyap.sh qc
```

1. **prescreen** (`openyap/prescreen.py`, S0). Word masks are the words ±0.1 s, with gaps under 0.3 s merged.
   Echo words are removed: a word is an echo word when the partner says the same normalised word (at least 4
   characters) between -0.3 and +1.5 s around it.
   * A conversation is rejected for any of:
     * transcript bleed: the identical-word lag histogram over -1..+2 s (20 ms bins, smoothed over 3) has
       excess > 0.01, peak-to-background > 3 and peak ≥ 10;
     * a `dropout` silence profile;
     * a track's DNSMOS OVR median below 2.5;
     * a noise floor above -40 dBFS.
   * A window (the 120 s grid from t = 0, full windows only) of a kept conversation is kept when:
     * speech covers at least 50% of it;
     * each speaker covers at least 5%;
     * each track's DNSMOS OVR median there is at least 2.3 (when at least 3 values exist).
   * Writes `screen/s0_conversations.json`, `s0_windows.tsv` and `s0_summary.json`.
2. **scan** (`openyap/scan_transcript_lags.py`). Computes the same transcript-bleed statistic over -10..+10 s.
   `summary.flagged_not_s0` of `screen/transcript_lags_wide.json` lists the conversations that S0 missed. They are
   rejected in the next stage but not used as controls.
3. **screen** (`openyap/screen_crosstalk.py run`, then `rescore`; S1). Measures crosstalk on the raw tracks.
   * Signal path:
     * decoding in 60 s blocks;
     * exact-zero runs of at least 20 ms;
     * clipped runs: at least 3 samples with |x| ≥ 32700;
     * resampling to 16 kHz with `resample_poly(1, 3)`;
     * a 150-3,800 Hz 4th-order band-pass;
     * frame energies over 20 ms windows with a 10 ms hop, floored at -110 dB.
   * Bleed model: track x holds g * y(t - lag) of its partner y. The lag is searched over ±3.5 s in 10 ms steps,
     per 10-minute block, on frames outside x's own speech. The echo-word window is -0.3..+3.5 s and x's own
     speech is guarded by 0.4 s.
   * Statistics:
     * BLR = p90 of x's energy where only the partner speaks, minus x's active level;
     * EXC = median energy where only the partner speaks, minus the median energy where nobody speaks;
     * both are measured per track and per window, at the conversation lag and at the window's block lag.
   * Integrity counts: zero runs inside the speaker's own word interiors (±30 ms), and clipped runs.
   * `rescore` applies the thresholds below together with the scan's list and writes
     `screen/s1_windows_decided_final.tsv`.
4. **restore** (`openyap/restore_inputs.py build|pending|verify` around `run/restore.sh`). Restores both tracks of
   every conversation with a kept window, plus the transcript-bleed conversations, which serve as controls for the
   next stage. Outputs go to `restored/openyap-1k/`.
5. **cut** (`openyap/cut_windows.py`; S2 and the source tars).
   * S2 recomputes BLR/EXC on the restored tracks at S1's lags. A window is rejected when either track at either
     lag has BLR ≥ -30 dB and EXC ≥ 3 dB.
   * The stage stops before writing tars (exit 3) when either:
     * S2 removes more than 3% of the windows that passed S1;
     * S2 catches fewer than 80% of the controls' S1-flagged windows (at least 10 needed).
   * `OY_ACCEPT_S2_STOP=1` writes the tars anyway.
   * Dev split: whole components of the speaker graph, smallest first, until at least 64 windows.
   * Train windows: blocks of at most 5 consecutive windows per conversation, shuffled with seed 20261012, each
     block given to the train tar with the fewest windows (128 train tars, plus 1 dev tar).
   * Record: `flac` is stereo 48 kHz PCM_16, 5,760,000 frames; channel 0 is the restored speaker a and channel 1
     the restored speaker b, with samples copied losslessly. `original_path.txt` holds the source paths and
     offset.
   * Writes `source/manifest.tsv`, `plan.json` and `summary.json`.
6. **produce**: the producer with writer name `openyap_24k_120s`, then the final check of the shards.
7. **qc** (`openyap/qc_shards.py headers|audio`). Checks the tar headers. On 300 random records it measures level,
   silence, 99% bandwidth, zero-run packets and the BLR of the final 24 kHz targets, and exports 20 records as WAV.

Thresholds (`openyap/thresholds.json`; `--decisions` or `OY_DECISIONS` replaces them):

| Key | Value | Rule |
| --- | --- | --- |
| `track_prominence` | 0.045 | track bleeds if prominence ≥ this and (BLR ≥ `track_blr_db` or EXC ≥ `track_exc_db`); the conversation is rejected |
| `track_blr_db` | -42.0 | |
| `track_exc_db` | 6.0 | |
| `window_blr_db` / `window_exc_db` | -35.0 / 3.0 | window rejected when a track has both at either lag (NaN passes) |
| `window_zero_runs_max` | 0 | zero runs inside own words allowed per track-window |
| `window_zero_runs_gated_max` | 1,000,000,000 | the same for noise-gated tracks (S0 silence profile `noise_gated`): the rule is off |
| `window_clip_runs_max` | 300 | clipped runs per track-window |
| `s2_window_blr_db` / `s2_window_exc_db` | -30.0 / 3.0 | S2 window rule on the restored tracks |
| `fallback_drop_aec_off` | false | (switch) reject every conversation with known AEC metadata |

## Basis Conversations 1500

Source: the Hugging Face dataset `basis-ai/basis-conversations-1500`, revision
`05a19e06b5eed44d701a8860b7e2336a44f762dd` (see "Licences and attribution": training is allowed, but the data and
anything derived from it must not be redistributed).

`BASIS_ROOT` is the downloaded release. It holds:

* `<lang>/segments.jsonl`: segments, with tracks by speaker, per-speaker on-call spans and bandwidth, muted spans
  and word-timed transcripts;
* `manifests/lhotse/recordings.jsonl`: segment lengths;
* `CHECKSUMS.sha256`;
* mono 48 kHz 16-bit FLAC tracks.

A segment has 2-28 per-speaker tracks on one clock. `BASIS_WORK` receives all outputs. Run the stages in this
order:

```
export BASIS_ROOT=... BASIS_WORK=...
bash run/basis.sh s0        # CPU, metadata only
bash run/basis.sh verify    # CPU: FLAC headers and sha256 checksums
bash run/basis.sh screen    # CPU, decodes every track of the ready segments
bash run/basis.sh dnsmos    # CPU, optional (DNSMOS_MODEL=sig_bak_ovr.onnx); not used by the shipped thresholds
bash run/basis.sh restore   # GPU
bash run/basis.sh cut       # CPU
NOISE_DIR=... bash run/basis.sh produce
bash run/basis.sh qc
```

1. **s0** (`basis/plan_pairs.py`). Picks windows and one pair of speakers per window.
   * Words are normalised with NFKC, casefolding, and keeping letters, marks and numbers. A long word has at least
     4 characters (2 in Japanese and Chinese, 3 in Korean). Masks pad words by 0.1 s and merge gaps under 0.3 s.
     The echo-word window is -0.3..+3.5 s, and it pools every other speaker of the segment.
   * A track is excluded from pairing (and from the windows' eligibility) when:
     * T0: it holds transcript copies of another track. The test runs the S0 lag statistic per ordered pair with
       at least 60 s of common on-call time, over ±3.5 s and over ±10 s (excess > 0.01, peak-to-background > 3,
       peak ≥ 10). Tracks share one clock, so the sign of the peak lag names the track holding the copies; within
       ±0.1 s both tracks hold them. Such tracks are restored as controls for S2.
     * T1: its bandwidth is below 3,000 Hz.
   * Windows lie on a 120 s grid from t = 0 on the segment clock. In each window, a speaker is eligible when they
     are on call for at least 119.5 s, their echo-free speech covers at least 5% (6 s), and they are not excluded.
   * The pair is the two eligible speakers with the most speech. Ties are broken by sha256 of
     `20261013|<segment>|<offset>|<speaker>`. Channel 0 is the member with the smaller sha256 of
     `20261013|order|<segment>|<offset>|<speaker>`.
   * A window is kept when it has at least 2 eligible speakers, no muted span overlaps it, and the pair's speech
     covers at least 50% of it. Other participants are not constrained.
   * Writes `screen/s0_segments.json`, `s0_windows.tsv`, `transcript_lags.json` and `s0_summary.json`.
2. **verify** (`basis/verify_audio.py headers|checksums`). Checks every present track's FLAC header against the
   expected length and its sha256 against `CHECKSUMS.sha256`. A segment is ready when all of its tracks verify:
   the screen needs every other track of the segment.
3. **screen** (`basis/screen.py features|run|clip-table|rescore`; S1). Uses the OpenYap signal path and statistics
   with the coherence diagnostic dropped, generalised to N tracks:
   * a lag search for every ordered pair with at least 60 s of common on-call time;
   * per-pair statistics that exclude frames where third speakers talk (at their own lags);
   * a pooled statistic over all other tracks;
   * window statistics against every track with at least 5 s on call in the window.

   A pair without a lag takes the lower median of the track's other lags (else 0). Frames where a track is off call
   or muted are ignored. The track rows carry the gate profile (exact-zero runs per hour on call) and the noise
   floor (p10 energy outside speech). `clip-table` adds the clipped-sample share of every track-window. `rescore`
   writes the decisions table `screen/s1_windows_decided_released.tsv`.
4. **dnsmos** (`basis/dnsmos_windows.py run|summarize`, optional). Scores DNSMOS P.835 OVR (non-personalised
   polynomial) on 37 clips of 9.01 s every 3 s per window, on the raw track resampled to 16 kHz with librosa.
   Clips with less than 10% speech are not scored, and a window median needs at least 3 clips at 25% coverage.
   Both DNSMOS thresholds are off in the shipped decisions, so the scores only fill the manifest's `ovr` columns.
5. **restore** (`basis/restore_inputs.py build|pending|verify` around `run/restore.sh`). Restores both tracks of
   every pair of a kept window, plus the controls, balanced by length into 16 parts. Only verified tracks are
   restored. Outputs go to `restored/basis-1500/<release path>`.
6. **cut** (`basis/cut.py`; S2 and the source tars).
   * S2 recomputes BLR/EXC of each restored pair track against every other track at S1's lags.
   * It stops before writing tars (exit 3) when it removes more than 3% of the windows that passed S1.
     `ACCEPT_S2_STOP=1` writes the tars anyway. The controls' hit rate is reported only.
   * Dev split: whole components of the participant graph (every participant of every segment of a conversation),
     smallest first, until at least 64 windows.
   * Train windows: blocks of at most 5 consecutive windows per segment, shuffled with seed 20261013, each block
     given to the train tar with the fewest windows (256 train tars, plus 1 dev tar).
   * Record: `flac` is stereo 48 kHz PCM_16, 5,760,000 frames, holding the restored pair in S0's channel order.
   * `ALLOW_MISSING` (a TSV with a `segment_id` column) lists segments to leave out, for example when their audio
     is not available. Without it, the stage refuses segments with kept windows that are not ready, screened or
     restored.
7. **produce / qc**: writer name `basis_24k_120s`. The audio report measures the BLR of the final 24 kHz targets
   against the partner, against the left-out participants, and pooled.

Thresholds (`basis/thresholds.json`; `DECISIONS` replaces them). Each window is decided per pair track x:

| Key | Value | Rule |
| --- | --- | --- |
| `track_prominence`, `track_blr_db`, `track_exc_db` | 0.07, -25.0, 6.0 | pair (x, y) bleeds if prominence ≥ 0.07 and (BLR ≥ -25 or EXC ≥ 6) ... |
| `track_lag_s` | [-0.1, 2.0] | ... counted only for peaks at lags (x hears y later when > 0) in this range |
| `track_level_mode` | "or" | BLR and EXC alternatively |
| `track_colocated_prominence`, `track_colocated_lag_s` | 0.3, 0.1 | a peak of prominence ≥ 0.3 within ±0.1 s flags the pair without level statistics |
| `floor_max_db` | -56.078125 | track excluded when its noise floor is above this |
| `window_blr_db`, `window_exc_db` | -35.0, 3.0 | window rejected when x has both against some y at either lag ... |
| `window_lag_s`, `window_prominence` | [-0.1, 2.0], 0.07 | ... where the row's lag lies in this range and its curve has this prominence |
| `window_pooled` | false | the pooled row is not used by the window rule |
| `window_zero_runs_max` | 0 | zero runs inside own words allowed per track-window ... |
| `window_zero_runs_gated_max`, `gate_runs_per_hour_min` | 1,000,000,000, 100 | ... and on gated tracks (≥ 100 zero runs per hour on call): rule off |
| `window_clip_runs_max` | null | clipped-run count not used |
| `window_clip_share_max` | 0.005 | at most 0.5% clipped samples per track-window |
| `dnsmos_track_min`, `dnsmos_window_min` | null, null | DNSMOS rules off (`dnsmos_clip_coverage` 0.25, `dnsmos_min_clips` 3) |
| `s2_window_blr_db`, `s2_window_exc_db` | -30.0, 3.0 | S2 window rule on the restored tracks |
| `fallback_no_excluded_on_call` | false | (switch) drop windows where another participant is on call |
| `s2_controls` | "report" | (switch) the controls' S2 hit rate does not stop the cut |

A track excluded by the track rule or the floor rule removes every window in which it is in the pair.

## Simulated conversations (SIM)

Source: LibriTTS-R (train-clean-100, train-clean-360, train-other-500), CC BY 4.0. The tracks are clean speech and
are not restored.

```
export SIM_WORK=... LIBRITTS_DIR=...
bash run/sim.sh pack        # optional: LIBRITTS_RELEASE=<extracted LibriTTS_R> -> $SIM_WORK/libritts_tars
bash run/sim.sh index
bash run/sim.sh dry-run
bash run/sim.sh simulate
bash run/sim.sh cut
NOISE_DIR=... bash run/sim.sh produce
bash run/sim.sh qc
```

1. **index** (`sim/index_libritts.py`). Reads `<subset>/data-NNNNN.tar.gz` with members
   `<spk>/<chapter>/<utt>.wav.audio.wav` (24 kHz PCM_16). `sim/pack_libritts.py` writes this layout from an
   extracted release; plans do not depend on how utterances are spread over tars.
   * Per utterance:
     * edge trim: frames above max(-70 dBFS, peak frame - 40 dB), keeping 50 ms;
     * active level and peak;
     * VAD runs: frames above active level - 24 dB, merged across gaps under 0.5 s.
   * Writes `index/index.tsv` and a PCM cache.
2. **turn-taking table**. `sim/turn_fit.json` holds 1,001-point quantile tables of turn length, within-turn pause,
   floor-transfer offset and interjection length, and the interjection rate. These are statistics of the word
   masks of the screened OpenYap-1k train windows (OpenYap-1k, CC BY 4.0, by The Agentic Data Company).
   `bash run/sim.sh fit` (`sim/fit_turns.py`) refits it from your own OpenYap screen outputs (`OY_MANIFEST`: the
   OpenYap `source/manifest.tsv`; `OY_FRAMES`: `screen/frames/`); then set `SIM_FIT`.
3. **dry-run / simulate** (`sim/simulate.py plan|render`, `sim/qc.py s0|s1`).
   * Planning:
     * Dev speakers come from train-clean-100/360 and are drawn with the seed until they hold twice the speech
       of the 72 dev records; train records never use them.
     * Records alternate overlap style (even n) and conversation style (odd n).
     * Each record pairs two speakers of its split, drawn in proportion to their remaining speech; the pairing
       uses no gender information.
     * Reuse: an utterance is used at most 4 times (`--max-uses`) and at most 2 times per style. It is never used
       twice in one record or twice with the same partner.
   * Overlap style: channel A starts at 0 s and channel B at U(0, 1) s; each is a continuous run of its speaker's
     utterances with U(0.2, 1.0) s pauses.
   * Conversation style: the floor alternates. Turn lengths, pauses and floor-transfer offsets are drawn from the
     table, and interjections arrive at the fitted rate (each at most 1.45 s, 0.35 s guard).
   * Levels: -26 dBFS active level ± U(3) dB per utterance; a track that would peak above 0.99 is scaled down.
   * Render: two mono 24 kHz tracks of exactly 120 s per record.
   * S0 checks:
     * format;
     * activity of each channel ≥ 5%;
     * speech in every 30 s quarter;
     * the reuse cap.
   * S1 checks:
     * the conversation-style medians against the table (±20%), with turn length and overlap share reported only;
     * overlap-style activity ≥ 0.85 and overlap share ≥ 0.7;
     * style balance per split and per train tar;
     * disjoint dev and train speakers;
     * the top-10 speaker share ≤ 0.073.
   * The target is 49,152 train records in 128 train tars (record n goes to tar (n // 2) % 128) and 72 dev records;
     a smaller speech pool yields fewer records.
4. **cut** (`sim/cut.py`). Upsamples both clean tracks 2x (`resample_poly(2, 1)`) to a stereo 48 kHz `flac`.
   `original_path.txt` is a JSON record of the speakers, utterances, offsets, gains and the plan hash. The stage
   refuses to run unless S0 and S1 passed (`ACCEPT_S1_STOP=1` proceeds after a failed S1).
5. **produce / qc**: writer name `sim_24k_120s`.

## The producer

`producer/` with `config/data/dialoguesidon2_producer.yaml`, run by `run/produce.sh SOURCE OUT WRITER` in 16 tasks.
Task NN processes part NN of the training source tars (`np.array_split` of the listing) with 8 data loader
workers, so each task needs at least 8 source tars. Task 0 also writes the validation shards.

1. Decode each source record. High-pass at 50 Hz. Peak-normalise each channel to 0.9, then the pair.
2. Cut into 120 s chunks keyed `<source key>_<start sample>`. Drop chunks in which neither channel exceeds 0.2.
   Pad the last chunk to 120 s.
3. Degrade each channel independently, with each step applied with probability 0.5, in this order:
   * reverberation: a pyroomacoustics shoebox with RT60 ~ U(0.1, 1.0) s, sides U(2, 20) m and random source and
     microphone positions; the clean target becomes the direct path;
   * additive noise from the noise shards at SNR ~ U(-5, 20) dB;
   * band limitation through 8, 16, 22.05, 24, 44.1 or 48 kHz;
   * quantile clipping (lower quantile U(0, 0.1), upper U(0.9, 1.0));
   * mp3 with qscale 10, 8, 4, 2 or 1;
   * packet loss: 3 packets of U(20, 200) ms per 10 s.

   Pad or cut both signals to 120 s again.
4. Resample to 24 kHz and 16 kHz. Peak-normalise every signal per channel, then jointly. Drop records that contain
   NaNs.
5. Mix the channels: `mixture = r * ch0 + (1 - r) * ch1` with r ~ U(0.3, 0.7), one r per record for all four
   mixtures (`producer/mixing.py`).

Before the tasks start, `run/produce.sh` runs `producer/preflight.py`. It checks the imports, pyroomacoustics and
the mp3 encoder, and passes one record through the whole pipeline to compare its layout with the expected one.
After each task, `producer/tasks.py check` compares the task's record count with the windows of its source tars
and scans its log for errors. When all 16 tasks are done, `finalize` makes the shards read-only and writes
`N_SHARDS.txt`.

The producer drops some windows by itself:

* chunks in which no channel exceeds 0.2;
* records that become NaN. Quantile clipping of a channel that is silent through most of the window can zero the
  whole channel, and the peak normalisation then divides by zero.

`MAX_SHORTFALL` allows that many missing windows per task.

### Shard format

`OUT/WRITER/train/tNN/worker-0-dataset-NNNNNN.tar` and `OUT/WRITER/valid/t00/...`. Keys are uuid4 hex. A shard
closes at 3 GB, about 31 records. Every record has 11 members (float32 unless noted; 120 s windows):

| Member | Content | Shape |
| --- | --- | --- |
| `input_wav.pth` | clean targets, 24 kHz (channel 0, channel 1) | [1, 2, 2,880,000] |
| `noisy_input_wav.pth` | degraded channels, 24 kHz | [1, 2, 2,880,000] |
| `noisy_input_wav16k.pth` | degraded channels, 16 kHz | [1, 2, 1,920,000] |
| `clean_mixture.pth` / `noisy_mixture.pth` | mixtures, 24 kHz | [1, 2,880,000] |
| `clean_16k_mixture.pth` / `noisy_16k_mixture.pth` | mixtures, 16 kHz | [1, 1,920,000] |
| `input_wav_lens.pth` | samples per target (int64) | [1] |
| `sr.index` | 24000 | |
| `sample_rate.pickle` | [24000] | |
| `names.pickle` | [`<source key>_<start sample>`] | |

### Differences from the current `src/sidon/data/preprocess` code

The released shards were produced before two fixes in the repository. The producer keeps the old behaviour by
default (`degradations: legacy`); `data.datamodule.degradations=upstream` selects the fixed versions instead:

* **packet loss**: the old version zeroes the samples in place. Clean and degraded signals start as views of one
  tensor, so when packet loss applies and none of the five earlier degradations of the channel did (probability
  1/64 per channel), the zeroed packets also appear in that channel's clean target.
* **length alignment**: the old band limitation does not restore the input length, and the old mp3 alignment can
  shorten the signal. Both are followed by padding/cutting to 120 s, so the degraded channel shifts or loses a few
  samples at the end.

Other differences from the current `dialogue_datamodule.py` and `preprocess.py`, kept as in the released data:

* chunks in which no channel exceeds 0.2 are dropped (the current code keeps every chunk);
* records with NaNs are dropped (the current dialogue pipeline does not filter them);
* chunk keys carry the start sample;
* RT60 is drawn up to 1.0 s (the current default is 2.0 s);
* the records hold no SSL feature inputs;
* the training shards are split into tasks, and only task 0 writes the validation shards.

## Differences from the runs that produced the data

The selection logic and parameters are unchanged. The checks listed under "Equivalence checks" below confirm this
on synthetic inputs. These things differ:

* **Restoration I/O**: the restorer reads its inputs with soundfile and writes 16-bit FLAC with explicit rounding.
  On synthetic files this is identical to the original environment's torchaudio/FFmpeg reading and writing.
* **CallHome restoration**: the CallHome part of the base root was restored with an earlier revision of the
  restoration script, which peak-normalised both channels jointly instead of each channel separately. That revision
  is not included; `common/sidon_restore.py` (the version used for Fisher, OpenYap-1k and Basis) normalises each
  channel.
* **Base-root validation**: the base root's validation shards came from a separate held-out dialogue set. Any
  held-out stereo source tars can take its place; they do not enter training.
* **OpenYap**: the scan's list of conversations with wide-range transcript copies is read from the scan output
  (`--extra-reject`) instead of being written into the code.
* **Basis**: the cut requires DNSMOS outputs only when a DNSMOS rule is set (the original always required them).
  With the shipped thresholds the DNSMOS stage can be skipped; the kept windows and the tar contents are the same,
  and only the manifest's `ovr_0`/`ovr_1` columns are empty.
* **SIM**: LibriTTS-R is read from tar.gz files in the layout above; `sim/pack_libritts.py` is new. Plans do not
  depend on how the utterances are spread over tars. The shipped `sim/turn_fit.json` holds the planner's tables
  and summary only, and gives the same plans as the full fit record.
* **Thresholds** ship as numbers in `openyap/thresholds.json` and `basis/thresholds.json`. The listening-based
  calibration that chose them is not included.
* **Stops and shortfalls** are overridden with plain switches (`*_ACCEPT_*_STOP=1`, `MAX_SHORTFALL`) instead of
  recorded notes.
* **Run records**: stage records no longer carry scheduler job ids, and some stage names, file names and messages
  are worded differently. The parameter hashes that guard resumption therefore differ from the original outputs,
  so start from empty work directories.

## Equivalence checks

Each stage was compared with the code that produced the released data. Both versions ran in separate processes on
the same synthetic inputs (synthetic speech-like tracks with injected bleed, synthetic transcripts, metadata and
release trees), and their outputs were compared, ignoring run-record fields.

| Stage | Result |
| --- | --- |
| `common/` crosstalk features and N-track statistics, layout, decisions, shard scans | identical |
| producer: each legacy transform forced on (packet loss on a shared tensor, band limitation, codec alignment, reverberation, clipping, chunking) | identical |
| producer: two seeded end-to-end runs of the data module on synthetic source and noise shards (3 s windows; 8 training and 2 validation records each, all degradation steps enabled at p = 0.5, mixing) | identical |
| producer: `produce.py` and `preflight.py` on synthetic shards | ran; record layout matches the expected member sizes |
| restoration: `denoise_speech` with stub TorchScript modules; input reading and 16-bit writing against the original environment | identical |
| restoration with the real Sidon v0.1 weights | not verified (CUDA TorchScript; no GPU used for this check) |
| base root: `common/pack_sources.py` | identical records apart from the random keys |
| OpenYap: S0, lag scan, S1 rows/frames/decisions (3 threshold sets x AEC fallback), restore lists and checks, S2, plan, manifest, tar members, shard headers | identical |
| OpenYap: audio QC command | not compared |
| Basis: S0, verification, S1 features/rows/clip table/decisions, DNSMOS clip selection and summary (stub scorer), restore lists and checks, S2, plan, manifest, tar members, target BLR | identical |
| Basis: DNSMOS scores with the real `sig_bak_ovr.onnx` on 37 synthetic clips | identical |
| Basis: shard headers/final check and audio QC end to end | not compared (thin wrappers over `common/shards_qc.py`) |
| SIM: index and PCM cache, plan (original code with the full fit record vs this code with `turn_fit.json`), rendering, S0/S1, cut tars, refit | identical |
| SIM: audio QC, shard headers | not compared |
| stage scripts in `run/` | no counterpart. `produce.sh` (one task), the first stages of `openyap.sh` and `sim.sh` were run on synthetic data; `basis.sh` and `restore.sh` were only syntax-checked |

## Tests

```
PYTHONPATH=src python -m unittest discover -s tests -p 'test_dialoguesidon2_data_*.py'
```

The tests use synthetic data only. They cover:

* the crosstalk screen;
* window cutting and the source-tar layout;
* the Basis pair choice and decisions;
* SIM planning and rendering with the shipped turn-taking table;
* shard checks;
* the producer's mixing, chunking and legacy degradations.

The producer tests that need pyroomacoustics are skipped when it cannot be loaded.

## Licences and attribution

* **OpenYap-1k**: The Agentic Data Company, `TheAgenticDataCompany/open-yap-1k` on the Hugging Face Hub, CC BY 4.0.
  `sim/turn_fit.json` holds statistics derived from its word timings and is shared under the same licence, with
  attribution.
* **Basis Conversations 1500**: Basis, `basis-ai/basis-conversations-1500`
  (https://huggingface.co/datasets/basis-ai/basis-conversations-1500), revision
  `05a19e06b5eed44d701a8860b7e2336a44f762dd`, licence `basis-data-license-1.0`. Read the licence before running
  the pipeline. Training on the data is permitted. The audio and material derived from it (restored tracks, screen
  tables, manifests, source tars and shards) must not be redistributed. If the licensor asks you to delete or stop
  using specific data, you must carry out that request yourself in every copy you hold. The tables and manifests
  keep the segment, speaker and conversation ids needed to find the affected windows.
* **LibriTTS-R**: Koizumi et al., 2023, CC BY 4.0.
* **CallHome and Fisher**: Linguistic Data Consortium; a licence from the LDC is required, and nothing derived from
  them is distributed here.
* **Sidon v0.1**: `sarulab-speech/sidon-v0.1` on the Hugging Face Hub, used for restoration.
* **DNSMOS P.835** (optional Basis step): Microsoft, `sig_bak_ovr.onnx` from the DNS Challenge repository.
