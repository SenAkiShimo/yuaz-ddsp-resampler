# AI Control Foundation Training

## Developer setup

Run:

```bash
./setup-ai-training.command
```

Recommended preset: Chinese Core. It downloads only the two Chinese GTSinger singers and the Breathy, Mixed_Voice_and_Falsetto, and Pharyngeal folders. The setup performs a Hugging Face dry-run first and prints the exact remaining size before confirmation.

The China route uses `hf-mirror.com` as a third-party/community mirror via `HF_ENDPOINT`; the official Hugging Face route remains selectable.

## Direct supervision

The v2 foundation uses only real GTSinger technique labels:

- breathy -> Breathiness
- falsetto -> Falsetto
- mixed_voice -> Mixed Voice
- pharyngeal -> Pharyngeal

Vibrato and Glissando are intentionally ignored by the DDSP timbre foundation for now. Tension and Voicing are not inferred from other technique labels.

The builder reads each Technique_Group JSON and aligns `ph_start` / `ph_end` plus the per-phoneme `breathy`, `falsetto`, `mix`, or `pharyngeal` labels to Yuaz DDSP frames. Supervised loss is applied only where the selected technique is actually annotated and the frame is voiced. If a legacy/missing JSON file has no annotation, the builder reports a fallback or rejects the pair instead of silently pretending every phoneme carries the technique.

## Native Yuaz feature space

Natural and Technique audio pass through the frozen current Yuaz encoder and decoder `extract_neural_ddsp_state()` path. The cached input state is Yuaz spectral envelope, AP, harmonic/noise gate and F0. Targets are paired residuals in log-envelope/AP-logit/gate-logit space.

```bash
./train-ai-control-foundation.command
```

The trainer is resumable. Final output:

```text
control_models/ai_control_foundation-v2.pt
```

Deep validates feature backend, checkpoint SHA-256 and the exact direct-control set before pinning a frozen copy into the AI voicebank generation.
