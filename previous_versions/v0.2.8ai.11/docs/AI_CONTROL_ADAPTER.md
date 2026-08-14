# AI Vocal Control Adapter — direct-technique v2

0.2.8ai uses one small temporal controller to manipulate Yuaz-native DDSP state. It never generates waveform audio.

Direct learned axes: Breathiness, Falsetto, Mixed Voice and Pharyngeal. They share one forward pass and output bounded residuals for spectral envelope, AP and harmonic/noise gate.

Tension and Voicing remain deterministic in this revision because GTSinger does not provide direct labels for those axes. Positive Breathiness is learned; negative Breathiness is a deterministic de-breathing fallback until a directly supervised opposite phonation dataset is added.

The training builder freezes the configured Yuaz checkpoint and extracts the same pre-synthesis DDSP state from natural and technique recordings. Model metadata binds the foundation to the Yuaz checkpoint SHA-256.

Training supervision is frame-local: GTSinger per-phoneme technique JSON is converted to control curves, and technique residual loss is masked to active voiced frames. This prevents silence, consonants, and untagged phonemes from teaching the control adapter a false whole-clip style shift.
