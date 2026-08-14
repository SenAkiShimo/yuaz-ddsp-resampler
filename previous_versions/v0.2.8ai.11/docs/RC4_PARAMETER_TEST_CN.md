# RC4.0 参数测试顺序

## 安装

如果要先彻底清掉 RC3.x **程序/runtime/wrapper/旧环境**，但保留声库训练状态：

```bash
./purge-previous-version.command
```

然后依次运行：

```bash
./setup-macos.command
./configure-macos.command
./install-openutau-macos.command
./doctor.command
```

`purge-previous-version.command` 不删除声库里的 `.yuaz-alpha8-rc3-3` / `.yuaz-alpha8-rc3-2`，因此不会故意抹掉已经训练好的声库。

## 先做自动对比测试

```bash
./vocal-controls-test.command
```

拖入一个已经 Prepare 的声库 WAV。脚本会输出 baseline、单参数正负变化和组合参数的 WAV，并打印 wall RTF。

## 第一轮建议听这些值

不要一开始直接拉到 ±100。先听：

- `YT-40 / YT40`：松 ↔ 紧/亮；
- `YB40 / YB70`：气声是否增加但仍保住咬字；
- `YV-40 / YV40`：周期/谐波扎实度；
- `YG-40 / YG40`：formant 上移 ↔ 下移，音高本身不应跟着变；
- `YO-40 / YO40`：收窄/偏暗 ↔ 更开、更有中频共鸣。

如果某个参数在 ±40 已经明显损伤音色或吐字，就不要继续扩大范围，应该回代码重新校准映射强度。

## Control Library

基础六参数不依赖外部训练库。要准备 GTSinger 等数据时：

```bash
./setup-control-datasets.command
./train-control-library.command
```

第二条命令会让你拖入已解压的 GTSinger 根目录，并生成：

```text
control_training/technique_profiles.npz
control_training/technique_profiles.json
```

RC4.0 暂时不把未经验证的 falsetto/mixed/belt/fry profile 自动接进正式渲染；先通过音色泄漏和 RTF 验证，再作为下一批 learned technique 参数接入。
