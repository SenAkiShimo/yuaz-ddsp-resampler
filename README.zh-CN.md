# Yuaz DDSP Resampler

Yuaz DDSP Resampler 是面向 OpenUtau 的 DDSP 重采样器。`0.2.8ai.13` 使用 24 kHz analysis/latent 与 48 kHz synthesis body，并加入宽频段斜率连续 crossover 和输出采样率感知的最高频保护。

English documentation: [README.md](README.md)

## 主要功能

- macOS 下的 OpenUtau 重采样器包装。
- 24 kHz 分析、48 kHz DDSP 合成。
- 高频频谱包络、aperiodicity 与 harmonic/noise 混合的频率相关扩展。
- 约 8.2–13.8 kHz 的宽 crossover，用于平滑衔接分析带宽边缘。
- 面向 44.1 kHz 输出的 harmonic ceiling 与 terminal guard。
- 可选 High-Band Foundation refinement 与声库高频 profile。
- Voicebank adapter、Fidelity Refiner、articulation preservation、响度归一化和可学习演唱控制。
- 带验证和回滚的声库状态迁移。

## 环境要求

- Apple Silicon Mac
- Python 3.14
- OpenUtau

项目使用固定依赖版本，详见 `requirements.lock.txt`。

## 安装

```bash
cd yuaz-ddsp-resampler-v0.2.8ai.13
chmod +x *.command scripts/*.command yuaz-ddsp-resampler

./setup-macos.command
./configure-macos.command
./self-test.command
./install-openutau-macos.command
./doctor.command
```

安装器会先验证并迁移兼容的声库状态，再清理旧的已安装 Yuaz runtime / wrapper / 已迁移 state container。源 WAV/OTO、训练数据集以及 `~/Documents/Yuaz-DDSP-Backups` 不会被删除。

## 高频诊断

在 OpenUtau 至少渲染一个音符后运行：

```bash
./highband-routing-diagnostic.command
```

当前 full-band backend：

```text
dual-rate-48k-ddsp-body-v3-slope-continuity-topguard
```

仓库根目录还提供多组 `*-test.command` 与 `*-diagnostic.command` 用于检查高频、Fidelity、咬字、控制参数和渲染状态。

## 训练与声库准备

兼容的已准备声库可以继续使用现有 adapter、Fidelity、articulation、high-band profile 和 Foundation 状态。需要重新准备或训练时，可使用仓库中的训练脚本。

相关文档：

- `docs/ARCHITECTURE.md`
- `docs/VOICEBANK_ADAPTATION.md`
- `docs/ARTICULATION_PRESERVATION.md`
- `docs/LEARNED_HIGHBAND.md`
- `HIGHBAND_FOUNDATION.md`
- `WEIGHTS.md`

## 仓库结构

```text
src/yuaz_ddsp_resampler/   Python runtime 与 DSP 实现
scripts/                   macOS 安装、迁移、诊断与训练脚本
control_models/            模型说明；训练后的权重不会提交到 Git
docs/                      架构与开发文档
previous_versions/         为兼容和参考保留的历史源码快照
```

历史工程记录和构建清单保存在 `docs/history/`，不参与 runtime。

## 开发检查

提交前建议运行：

```bash
python3 -m compileall -q src/yuaz_ddsp_resampler
./self-test.command

for f in *.command scripts/*.command; do
  bash -n "$f"
done
```

仓库与发布说明见 `CONTRIBUTING.md` 和 `docs/GITHUB_SETUP.md`。

## 权重与数据集

训练权重默认不进入 Git。数据集和衍生权重可能受各自来源许可约束；公开分发前请阅读 `WEIGHTS.md` 与 `THIRD_PARTY_NOTICES.md`。

## License

见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
