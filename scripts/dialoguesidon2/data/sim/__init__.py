"""Simulated two-speaker conversations from LibriTTS-R (see the README in the parent directory).

Modules: base (parameters, keys, the energy VAD, activity masks and the turn segmentation), index_libritts (the
utterance index and its PCM cache), pack_libritts (packs a LibriTTS-R release into the tar.gz layout the index
reads), fit_turns (the turn-taking fit on screened OpenYap-1k word masks), simulate (plan and render), qc (checks of
the clean tracks and the producer output) and cut (stereo 48 kHz source shards for the producer).
"""
