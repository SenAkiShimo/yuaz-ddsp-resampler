# Voicebank preparation

`prepare-voicebank.command` provides four modes.

## Fresh Fast Profile

Builds voicebank metadata, articulation data, high-band profiles, loudness statistics, and registry entries without deep gradient adaptation.

## Clean Deep Retrain

Creates a backup, removes the current release state, then trains from scratch.

Deep adaptation uses two stages. Stage A trains identity-related adaptation and multipitch routing. Stage B freezes identity and routing parameters and calibrates a limited spectral/AP parameter set against source-relative band balance and harmonic contrast. A holdout set selects the best Stage-B checkpoint; the Stage-A state remains available as a fallback.

## Continue Deep Adapt

Backs up the current state, continues deep adaptation, and reruns the second-stage calibration.

## Relearn High-Band

Backs up the current state and forces source-WAV re-analysis for High-Band v3 without retraining the main adapter.

## Backups

Backups are stored under `~/Documents/Yuaz-DDSP-Backups/<voicebank>/`. Analysis caches are rebuildable and are not required to restore runtime sound.
