# Yuaz DDSP Resampler

一个基于 Yuaz SGR 的 OpenUtau 外部重采样器。项目包含声库准备、多音阶音色适配、咬字轨迹保留、可选的学习式高频扩展、响度归一化以及 OpenUtau 安装脚本。

## 环境要求

- macOS
- Python 3
- OpenUtau
- 本地 Yuaz SGR 仓库
- Yuaz SGR checkpoint

本仓库不包含 Yuaz SGR 源码或模型权重。相关说明见 [UPSTREAM.md](UPSTREAM.md) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装环境

```bash
chmod +x *.command scripts/*.command yuaz-ddsp-resampler
./setup-macos.command
./configure-macos.command
```

`setup-macos.command` 会建立项目独立的 `.venv`。默认使用清华 PyPI 镜像，失败时自动回退到默认源。

`configure-macos.command` 会读取本地 Yuaz SGR 仓库与 checkpoint；默认路径不存在时会要求手动拖入。

## 准备声库

```bash
./prepare-voicebank.command
```

可选模式：

1. Fresh Fast Profile
2. Clean Deep Retrain
3. Continue Deep Adapt
4. Relearn High-Band

本版本训练状态保存在：

```text
<voicebank>/.yuaz-alpha8-rc3-2/
```

任何会修改训练状态的 Prepare 模式都会先建立外部备份。备份目录：

```text
~/Documents/Yuaz-DDSP-Backups/<voicebank>/
```

也可以手动执行：

```bash
./backup-training.command
./restore-previous-training.command
./list-training-backups.command
```

Clean Deep 使用两阶段训练。第一阶段完成声库身份与多音阶路由适配；第二阶段固定身份相关参数，仅校准主体频段的谱平衡、谐波峰谷与非周期成分。第二阶段带独立验证与 checkpoint 回退。

## 安装到 OpenUtau

```bash
./self-test.command
./install-openutau-macos.command
```

重启 OpenUtau 后选择：

```text
Yuaz-DDSP-Resampler-v0.2.7-alpha.8-rc.3.2.sh
```

卸载当前 OpenUtau 条目：

```bash
./uninstall-openutau-macos.command
```

## 参数

| Flag | OpenUtau 参数 | 范围 | 默认 |
|---|---|---:|---:|
| `YM` | Yuaz Timbre Morph | -100..100 | 0 |
| `YD` | Yuaz Learned Detail | -100..100 | 0 |
| `YH` | Yuaz High-Band | 0 或 80..120 | 0 |

`YH0` 关闭学习式高频扩展。非零 `YH` 按百 Hz 表示 Yuaz-only crossover。

详细说明见 [docs/CONTROLS.md](docs/CONTROLS.md)。

## 检查声库状态

```bash
./inspect-voicebank.command
```

## 响度设置

```bash
./loudness-settings.command
```

## 许可证

本仓库源码使用 MIT License。Yuaz SGR 源码、checkpoint、数据集及其他第三方内容适用各自的许可条款。
