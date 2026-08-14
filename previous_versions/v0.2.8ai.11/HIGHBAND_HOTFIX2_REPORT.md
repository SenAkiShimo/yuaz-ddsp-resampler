# High-Band Hotfix 2

Hotfix 1 restored the source-texture continuity branch alongside the neural Foundation, but the 24 kHz Yuaz body itself remained untouched. Because that body terminates around its Nyquist edge, its horizontal spectral roof remained visible even when energy was added above it.

Hotfix 2 changes the runtime to a complementary Nyquist crossover:

- source-texture bridge begins below the body edge;
- the body edge is gently tapered only inside the overlap region at high YH strength;
- 12–15 kHz and 15–18 kHz learned/neural bands receive conservative energy floors derived from the actual 8–11 kHz edge;
- the upper floor scales existing learned/neural texture rather than adding a flat broadband ceiling;
- global high-band safety limiting remains active.

No adapter, Fidelity, Deep, or Foundation retraining is required. Existing Foundation r1/r2 checkpoints remain compatible.
