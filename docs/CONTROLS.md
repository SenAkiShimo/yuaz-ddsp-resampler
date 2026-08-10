# Controls

## YM — Timbre Morph

Range: `-100..100`, default `0`.

YM shifts the trained multipitch timbre-routing target while leaving the requested F0 unchanged. The full range corresponds to approximately `-12..+12` semitones in routing space.

## YD — Learned Detail

Range: `-100..100`, default `0`.

YD scales the trained detail paths. Negative values reduce learned detail; positive values increase it.

## YH — High-Band

Range exposed by OpenUtau: `0..120`, default `0`.

`YH0` disables the learned high-band extension. Non-zero values are clamped to `80..120` and represent the Yuaz-only crossover in hundreds of hertz. For example, `YH100` corresponds to a 10 kHz crossover setting.
