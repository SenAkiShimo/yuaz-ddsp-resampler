# Learned High-Band Restoration

The main Yuaz/DDSP body runs at 24 kHz, so its Nyquist limit is 12 kHz. Output resampling to 44.1 kHz cannot recreate information above that limit by itself.

`YH` is therefore a post-DDSP restoration amount:

- `YH0`: disabled
- `YH25`: light restoration
- `YH50`: medium restoration
- `YH100`: full restoration

The runtime uses the prepared source high-band profile where it is reliable, but it no longer allows an almost-zero 13–20 kHz profile to force the entire upper spectrum to remain blank. It anchors the extension to the rendered 6.5–12 kHz edge, applies a conservative reconstruction floor, and synthesizes bounded harmonic/noise content above the DDSP Nyquist edge.

Use `highband-nyquist-diagnostic.command` to measure the rendered 8–12, 12–13, 13–16 and 16–20 kHz bands.
