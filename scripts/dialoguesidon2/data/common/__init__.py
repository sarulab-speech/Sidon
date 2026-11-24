"""Corpus-independent helpers of the DialogueSidon2 data pipelines.

Modules:

* core: constants of the 48 kHz / 100 fps / 120 s grid, run records, atomic writes, FLAC headers, TSV and masks;
* crosstalk: the band-passed energy features and the bleed (crosstalk) statistics of per-speaker tracks;
* decisions: the track, window and integrity rules applied to those statistics;
* layout: the speaker-disjoint dev split, blocks of consecutive windows, the seeded tar assignment and tar writing;
* restore: restoration input trees and output checks;
* sidon_restore: restoration with the public Sidon v0.1 model (sarulab-speech/sidon-v0.1);
* shards_qc: checks of the producer's output shards.

Entry points live in the sibling directories (openyap/, basis/, sim/, producer/); each puts this directory's parent
on sys.path, so the modules are imported as `common.<name>`.
"""
