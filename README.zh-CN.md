# Yuaz DDSP Resampler v0.2.8ai.14

面向 macOS / OpenUtau 的 sample-conditioned Yuaz/DDSP 重采样器。

v0.2.8ai.14 在保留 48 kHz 合成主体、高频连续性路由与 output-rate top-band guard 的基础上，加入了兼容 Yuaz checkpoint 的底模注册系统，并进一步隔离不同版本与不同底模对应的声库训练状态。

## 主要变化

- 不再依赖固定的 `checkpoint_300k.pt` 文件名，可检查并导入结构兼容的 Yuaz `.pt`。
- 导入时仅提取重采样器需要的 Encoder、DDSP Decoder 与 RVQ 权重，生成紧凑 runtime checkpoint。
- 记录原始 checkpoint 的 SHA-256 与训练 step。
- 每个 ai.14 Deep generation 记录底模来源；当前底模与训练时不一致时，渲染会拒绝载入该 learned state。
- 与 v0.2.8ai.13 并存安装，不覆盖旧 runtime、OpenUtau wrapper 或声库状态。
- 使用独立端口、状态命名空间、训练产物文件名和缓存目录。
- 本版本禁用 destructive predecessor purge。

详细设计见 [`docs/BASE_MODEL_REGISTRY.md`](docs/BASE_MODEL_REGISTRY.md) 与 [`docs/SIDE_BY_SIDE_SAFETY.md`](docs/SIDE_BY_SIDE_SAFETY.md)。

## 命令入口

日常调用统一通过一个 launcher；真正实现仍集中在 `scripts/`。

```bash
./commands/run.command list
./commands/run.command find yv
./commands/run.command doctor
```

完整说明与兼容别名见 [`commands/README.md`](commands/README.md)。

## 模型权重

**本仓库不包含 Yuaz checkpoint 或其他训练得到的 `.pt` 权重。** 请从具有授权的来源取得兼容 checkpoint，然后在本地导入：

```bash
./commands/run.command probe-yuaz-checkpoint
./commands/run.command import-yuaz-checkpoint
./commands/run.command list-yuaz-checkpoints
./commands/run.command select-yuaz-checkpoint
```

导入器会先验证 Encoder / DDSP Decoder / RVQ 的结构覆盖率，再注册本地 runtime。完整训练 checkpoint 中可能还包含生成器、判别器、优化器和 scaler 等状态；这些内容不属于 OpenUtau resampler 的运行时依赖。

权重分发与来源说明见 [`WEIGHTS.md`](WEIGHTS.md)。

## 安装

```bash
chmod +x commands/run.command yuaz-ddsp-resampler
./commands/run.command setup-macos
./commands/run.command configure-macos
./commands/run.command self-test
./commands/run.command install-openutau-macos
./commands/run.command doctor
```

配置时可以提供完整兼容 checkpoint，也可以提供此前由 importer 生成的紧凑 runtime checkpoint。

## 声库准备

```bash
./commands/run.command deep-train-voicebank
```

v0.2.8ai.14 只写入 `.yuaz-0.2.8ai14`，不会迁移、重命名、覆盖或删除 `.yuaz-0.2.8ai13`。

ai.14 的主要训练产物采用独立文件名：

```text
adapter.ai14.pt
timbre_profiles.ai14.pt
training.ai14.json
fidelity_refiner.ai14.pt
fidelity_training.ai14.json
deep_validation.ai14.json
highband_profiles_v3.ai14.json
cache_ai14/
highband_cache_v3_ai14/
```

## 与 ai.13 并存

v0.2.8ai.14 使用 TCP 端口 `47886`；v0.2.8ai.13 保持原有端口。两个 OpenUtau resampler wrapper 可以同时保留。

`purge-previous-version` 在 v0.2.8ai.14 中被明确禁用。

## 上游项目

Yuaz DDSP Resampler 使用 Yuaz SGR 的 Encoder 与 DDSP Decoder 架构。上游关系见 [`UPSTREAM.md`](UPSTREAM.md)。
