# 0.2.8ai.11 High-Band continuity hotfix 1

- Fix the YH100 “hard wall” / sparse-upper-band failure seen after pinning High-Band Foundation v1.
- Keep Foundation v1 checkpoints fully compatible, but no longer let Foundation bypass the voicebank source-texture continuity path.
- Add an adaptive hybrid path: Foundation supplies learned high-band events; the voicebank source-texture branch bridges roughly 8–14 kHz and fills 12–20 kHz only where Foundation coverage is locally weak.
- Add temporal-coverage diagnostics and a sparse-Foundation regression to the project self-test.
- Add Foundation v2 training: runtime-matched residual target, multi-resolution high-band magnitude loss, framewise 9.5–22 kHz band-envelope loss, light waveform loss, and a wider receptive field.
- `train-highband-foundation.command` now writes `highband_foundation-v2.pt`; runtime and `learn-highband.command` prefer v2 but fall back to existing v1 weights.
- No Deep/adaptor/fidelity retraining is required to test the runtime hotfix with an already pinned v1 Foundation.

# 0.2.8ai.11

- Freeze the 0.2.8ai.9 manual high-band fallback.
- Add High-Band Foundation v1 bandwidth audit, paired-shard preparation, training and probing.
- Use existing GTSinger/VocalSet/Phonation Modes data; no new download is required for the first experiment.
- Oversample low-F0 material and track low-F0 validation separately.
- Add optional runtime `highband_foundation.pt`; YH falls back to the previous voicebank-profile path if absent.
- Keep the neural residual hard-masked to the upper band.
- Pin a trained foundation with `learn-highband.command` without repeating full Deep.
- Migrate 0.2.8ai.9 state into `.yuaz-0.2.8ai11` before previous-version purge.

# 0.2.8ai.11

- Replaced the v0.2.8ai.8 upper-band sine-ladder reconstruction with a source-texture nonlinear exciter. The new path derives 12–20 kHz content from the rendered voice itself, so high-band phase, articulation and spectral motion follow the note instead of appearing as perfectly stable synthetic lines.
- Removed the fixed high-partial ceiling and harmonic-index attenuation that could leave low notes with little or no content above roughly 12–14 kHz.
- Added band-limited quadratic/cubic excitation, deterministic narrow spectral skirts, profile-shaped spectral tilt and low-pitch compensation.
- Keeps the voicebank's own learned High-Band profile as the timbre/envelope constraint; no external singer data is mixed into runtime restoration.
- Preserves Raw WAV bypass, all learned-control packs, Deep/Fidelity/articulation/loudness state and v0.2.8ai.8 High-Band profiles during migration.
- Installation migrates the validated v0.2.8ai.8 state into `.yuaz-0.2.8ai11`, validates it, then removes previous installed Yuaz versions/states. Source WAV/OTO, datasets and `~/Documents/Yuaz-DDSP-Backups` are preserved.
- Runtime identity: 0.2.8ai.11, port 47885, registry `voicebank_registry-0.2.8ai11.json`.
