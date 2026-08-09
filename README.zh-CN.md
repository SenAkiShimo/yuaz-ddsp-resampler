# Yuaz DDSP Resampler v0.2.7-alpha.1

面向 UTAU / OpenUtau 的 sample-conditioned DDSP 重采样器实验版本，基于本地 Yuaz SGR Encoder/DDSP 运行。

本版本重点解决两类已经在实际多音阶声库中观察到的问题：

- 咬字轨迹虽然保留，但会把当前录音音阶的一部分音色一起带入，形成轻微 timbre leakage；
- 咬字轨迹中的宽带 spectral tilt 会把 3–9 kHz 一起压暗，使声音重新出现闷感。

v0.2.7-alpha.1 不更换 DDSP、Adapter、Fidelity Refiner 或严格响度归一化。它只重写“有声咬字轨迹”的来源。

## Canonical Articulation

对于 UTAU multipitch 声库，程序会利用 `prefix.map` / subbank 信息，把同一个 `base_alias` 在不同真实音阶中的咬字对齐：

```text
G3 zhi ─┐
D4 zhi ─┤
G4 zhi ─┼─> timbre-neutral trajectory ─> canonical zhi articulation
D5 zhi ─┘
```

共同的时间变化更可能代表声库自己的发音方式，而不同音阶之间的整体亮暗、声区和音色差异交给 subbank timbre prototype 负责。

程序会为声库生成：

```text
Voicebank/.yuaz/
└── articulation/
    ├── index.json
    └── canonical/
        ├── <hash>.npz
        ├── <hash>.npz
        └── ...
```

`index.json` 会记录：

- canonical alias 数量；
- 其中多少来自真实 multipitch 对齐；
- 多少只能使用单音阶 neutral fallback；
- 每个 canonical trajectory 的 subbank 数量和 coherence。

## Timbre-neutral trajectory

有声咬字区不复制原始 waveform，也不恢复 TD-PSOLA。

流程仍然保持单周期源：

```text
原 WAV 无声辅音 / 瞬态
        ↓
100% 原波形
        ↓
可靠 voiced onset
        ↓
Canonical articulation trajectory
        ↓
唯一一份 target-F0 DDSP waveform
        ↓
Fidelity Refiner
        ↓
严格 final-render loudness normalization
```

Canonical trajectory 会主动削弱：

- 静态 spectral tilt；
- 宽带音色差异；
- 当前 subbank 的整体亮暗。

同时保留：

- formant 随时间的运动；
- 局部谱峰/谱谷变化；
- 元音建立轨迹；
- articulation energy trajectory。

另外加入 clarity guard：咬字轨迹不能把约 3–9 kHz 作为一个整体长期大幅压低。

## 单音阶 / 缺失 multipitch alias

如果一个 `base_alias` 只存在一个可靠 subbank，程序仍会建立 `single_neutral_fallback`。

它不会直接使用原始绝对频谱，而会先去掉宽带 timbre/tilt 后再保存，因此比 v0.2.6 的逐样本 trajectory 更不容易携带音色。

## 严格响度归一化保持不变

v0.2.7 沿用 v0.2.6-alpha.2 的 final-render normalization：

```text
Hybrid + Fidelity 最终波形
        ↓
active RMS
        ↓
默认 -18.0 dBFS
        ↓
soft peak guard
        ↓
再次校准 active RMS
        ↓
OpenUtau VOL / DYN
```

默认：

- target active RMS：-18.0 dBFS
- peak ceiling：-1.0 dBFS
- normalization tolerance：约 ±0.05 dB

用户在 OpenUtau 中的 VOL / DYN 仍然在内部归一化之后生效。

## 从 v0.2.6-alpha.2 升级

解压本版本后：

```bash
cd ~/Downloads/yuaz-ddsp-resampler-v0.2.7-alpha.1
chmod +x *.command scripts/*.command yuaz-ddsp-resampler
./purge-previous-version.command
./setup-macos.command
./configure-macos.command
./self-test.command
```

`purge-previous-version.command` 会清理旧程序、旧 OpenUtau resampler 和旧后台，但不会删除：

- 声库中的 `.yuaz/`；
- 本地 Yuaz SGR 仓库；
- Yuaz checkpoint。

## 已经训练好的声库：请选择 Fast Profile

本版本没有新的 Adapter / Refiner 梯度训练目标。

已经完成 Deep Adapt 的声库不要重新 Deep Adapt。运行：

```bash
./prepare-voicebank.command
```

拖入声库根目录，然后选择：

```text
1) Fast Profile
```

Fast Profile 会：

- 复用已有 `.yuaz/cache/`；
- 保留 Adapter / Anti-Leak / timbre prototypes / Fidelity Refiner；
- 重建 UTAU subbank routing；
- 建立 canonical articulation dictionary；
- 刷新 strict loudness profile；
- 刷新全局 voicebank registry。

第一次建立 canonical dictionary 需要读取已有缓存并做短时频谱分析，因此会比普通 registry refresh 更慢，但不会重新跑 Yuaz Encoder。

## 检查声库

```bash
./inspect-voicebank.command
```

新增摘要：

```text
Canonical articulation:
  strategy: multipitch_common_trajectory_with_timbre_neutral_fallback
  alias_count: ...
  multipitch_canonical_count: ...
  single_neutral_fallback_count: ...
  mean_coherence: ...
  clarity_guard: 3-9kHz broad attenuation floor
```

对于 multipitch 声库，`multipitch_canonical_count` 应该明显大于 0。

## 安装到 OpenUtau

```bash
./install-openutau-macos.command
```

完全退出并重新启动 OpenUtau，选择：

```text
Yuaz-DDSP-Resampler-v0.2.7-alpha.1.sh
```

默认后台端口：

```text
47860
```

## 本版本最值得比较的听感

建议与 v0.2.6-alpha.2 比较：

1. `zhi / shi / chi / si` 等咬字是否仍保留声库自己的口音；
2. 咬字瞬间是否少了一点“突然换音色”的感觉；
3. 元音入口和稳定元音是否少闷；
4. multipitch 切换时身份是否更连续；
5. strict normalization 是否保持原有表现。

如果 canonical trajectory 不存在，运行时会自动退回 `neutralized_local`，不会导致该 alias 无法渲染。

## 第三方组件

本仓库不包含 Yuaz SGR 源码或 checkpoint。使用者需要自行提供本地 Yuaz SGR checkout 与模型文件。详见 `THIRD_PARTY_NOTICES.md` 和 `UPSTREAM.md`。
