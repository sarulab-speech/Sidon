"""The producer of the DialogueSidon2 training shards: 120 s stereo source windows -> records of 11 members with
per-channel degradations and a mixture (see the README of scripts/dialoguesidon2/data).

* degradations: the degradation functions exactly as they produced the released data, next to the upstream
  versions that later fixed two defects (selected with the data module's `degradations` setting);
* mixing: the mixture of the two channels;
* datamodule: DialogueProducerDataModule, the Hydra target of config/data/dialoguesidon2_producer.yaml;
* produce: the entry point that writes one producer task's shards;
* tasks: expected record counts, per-task checks and the final check;
* preflight: environment checks and a dry run of one record.
"""
