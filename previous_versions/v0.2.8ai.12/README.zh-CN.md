# Yuaz DDSP Resampler 0.2.8ai.11

## High-Band Foundation 高频连续性修复（v2 / v1 兼容）

这一版冻结 0.2.8ai.9 的现有声音与 12 个 OpenUtau 参数，不再继续手工堆高频谐波算法。High-Band Foundation 仍负责用真实全带宽歌声学习 24 kHz 带宽受限输入缺失的 12–22 kHz；这一 hotfix 额外修复了 Foundation v1 在实际 YH100 下“只偶尔冒出高频、9–12 kHz 仍像一刀斩”的问题。现有 v1 权重不作废，安装 hotfix 后即可先测试。

现有参数仍为：

`YM YD YH YT YB YV YG YO YF YX YP YR`

- `YR1`：Raw WAV bypass。
- `YH0`：关闭高频恢复。
- `YH100`：Foundation 与声库 source-texture continuity 混合恢复；Foundation 缺失时自动使用声库 profile 高带宽方案。

## 为什么改成训练模型

Yuaz 主体仍以 24 kHz 工作，因此主体 Nyquist 上限是 12 kHz。0.2.8ai.8/0.2.8ai.9 已经证明：规则谐波可以把 >12 kHz “造出来”，但容易出现过分整齐的谐波梯子；改用 source-texture 后又容易在低音和 13 kHz 以上衰减过快。

原版 0.2.8ai.11 把 Foundation 固定进声库后会绕开 source-texture fallback；这正是高频稀疏时暴露截止墙的关键问题。hotfix 改为：

```text
High-Band Foundation
→ 学会一般人声的 12–22 kHz 应该怎样存在

voicebank highband_profiles_v3.json + source texture
→ 约束声库自己的高频倾斜/风格，并在 Foundation 局部掉空时提供连续桥接
```

Foundation 输入是同一段歌声人为通过 24 kHz 瓶颈后的版本，target 是原始全带宽录音。Foundation 仍只输出被限制在高频的 residual；continuity 分支由已经渲染出的 Yuaz 声音自身派生，并从约 8.2 kHz 开始做软桥接，因此不会在 9.5 kHz mask 边缘留下明显“墙”。

## 第一轮不下载新训练集

默认审计本机已经存在的：

- GTSinger Chinese Core
- VocalSetMirror
- PhonationModesOSF

MOCHA 不参与 High-Band Foundation，因为它用于 YV/YO 的 articulatory/glottal 监督，不提供目标 12–22 kHz 全带宽信息。

### 1. 安装 0.2.8ai.11

```bash
./setup-macos.command
./configure-macos.command
./self-test.command
./install-openutau-macos.command
./doctor.command
```

安装会先迁移最新的 0.2.8ai.9 prepared state 到 `.yuaz-0.2.8ai11`，验证成功后再删除之前安装的 Yuaz runtime / wrapper / 已迁移旧 state。源 WAV、OTO、数据集和 `~/Documents/Yuaz-DDSP-Backups` 不删除。

### 2. 审计现有数据的真实有效高频

```bash
./audit-highband-datasets.command
```

输出：

```text
~/YuazControlDatasets/HighBandFoundation/audit.json
```

审计会记录：采样率、Nyquist、10–12 / 12–16 / 16–20 / 20–22 kHz 相对能量、估计 F0，并排除名义采样率很高但上部频带实际上为空的录音。

VocalSet Parquet 中真正通过审计的音频才会保留为本地 FLAC cache；失败项会删除临时 cache。

### 3. 制造 paired training shards

```bash
./prepare-highband-training.command
```

默认：

```text
train: 6000 × 1 s
val:    800 × 1 s
```

每个 target 都是真实 48 kHz 全带宽片段；input 会经过：

```text
48 kHz target
↓
降到 24 kHz
↓
再升回 48 kHz
↓
得到 Yuaz 类似的 12 kHz Nyquist 缺失输入
```

低 F0（尤其 <160 Hz）会自动提高抽样权重，validation 也单独统计 low-F0 loss。

如果要更大的第一轮：

```bash
YUAZ_HIGHBAND_SEGMENTS=12000 \
YUAZ_HIGHBAND_VAL_SEGMENTS=1500 \
./prepare-highband-training.command
```

### 4. 训练 PT

```bash
./train-highband-foundation.command
```

默认 10 epochs，自动优先 MPS。输出：

```text
control_models/highband_foundation-v2.pt
```

并备份到：

```text
~/Documents/Yuaz-DDSP-Backups/control-models/0.2.8ai.11/highband_foundation-v2.pt
```

该 PT 是独立 bandwidth-extension foundation，不覆盖现有四只 learned-control PT。

### 5. 检查模型和 CPU 开销

```bash
./probe-highband-foundation.command
```

会打印模型 metadata、参数量以及 1 秒音频的 model-only CPU RTF。

### 6. 固定到声库

训练完成后，对每个需要测试的声库运行：

```bash
./learn-highband.command
```

这一步不重新 Deep。它会：

1. 复制当前声库 generation；
2. 重新整理该声库自己的 `highband_profiles_v3.json`；
3. 把 `highband_foundation-v2.pt` 冻结复制成 `highband_foundation.pt`；
4. 验证后切换 ACTIVE。

随后在 OpenUtau 中测试 `YH0 / YH25 / YH50 / YH100`。

## High-Band Foundation v2 架构与 v1 兼容

v2 仍是轻量 waveform residual BWE network，但使用更宽的 dilated receptive field，并把训练目标改为以高频幅度/时间包络为主：

- 输入：带宽受限波形 + F0 条件；
- v2 为 92,681 参数；旧 v1 约 4.5 万参数仍可加载；
- 训练采样率固定 48 kHz；
- 输出仅作为 high-band residual；
- 运行时 crossover 约 9.5 kHz，约 12.1 kHz 后完全由 foundation 补全；
- 上限约 22 kHz；
- 44.1 kHz OpenUtau 输出会在模型内部临时转成 48 kHz 推理，再返回原输出采样率；
- 声库 profile/source-texture 还承担连续性 floor：Foundation 弱的帧会补空洞，Foundation 强的帧尽量不覆盖。

如果 foundation 尚不存在，YH 会保留 0.2.8ai.9 fallback，因此这一版在训练前仍可正常渲染。


### 先不用重训也能测试 hotfix

如果当前声库已经固定过 v1 `highband_foundation.pt`，安装这个 hotfix 后直接渲染 `YH100` 即可测试新的 hybrid continuity。**不需要先 Deep，也不需要先重新训练 Foundation。**

确认路由：

```bash
./highband-routing-diagnostic.command
```

重点看：`highband_continuity_hybrid_used`、`highband_temporal_coverage_before`、`highband_temporal_coverage_after`。

如果 runtime hotfix 的方向正确，再运行 `./train-highband-foundation.command` 训练 v2，并对每个声库执行 `./learn-highband.command` 固定新权重。

## 权重许可

源码许可和训练权重许可必须分开处理。`highband_foundation-v2.pt` 的可分发条件取决于实际进入 shards 的源数据集。训练 metadata 会保留 provenance/rights 提示；正式 GitHub v1 前应单独整理 `WEIGHTS.md`，不要把 derived PT 默认当作源码 LICENSE 覆盖。
