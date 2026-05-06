# 华为昇腾 NPU Profiling 使用指南 (torch_npu.profiler)

## 一、基础配置

### 1.1 轻量级采集 (Level0, 最小开销)

```python
import torch_npu

experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("<输出目录>"),
    record_shapes=False,
    profile_memory=False,
    with_stack=True,
    with_modules=True,
    with_flops=False,
    experimental_config=experimental_config
)
```

### 1.2 详细采集 (Level1 + 内存分析)

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("<输出目录>"),
    record_shapes=False,
    profile_memory=True,
    with_stack=True,
    with_modules=True,
    with_flops=False,
    experimental_config=experimental_config
)
```

### 1.3 完整采集 (Level2 + 硬件指标)

```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=True,
    op_attr=True,
    data_simplification=False,
    record_op_args=True,
    gc_detect_threshold=None
)

prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("<输出目录>"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_modules=True,
    with_flops=True,
    experimental_config=experimental_config
)
```

## 二、参数说明

### 2.1 ExperimentalConfig 参数

| 参数                    | 说明                                                                                    | 推荐值               |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------- |
| `export_type`         | 输出格式，`Text` 为文本格式                                                                     | `ExportType.Text` |
| `profiler_level`      | 采集深度：Level0(轻量) / Level1(详细) / Level2(完整)                                             | 按需选择              |
| `msprof_tx`           | 是否采集 msprof 事务数据                                                                      | `False`           |
| `aic_metrics`         | AI Core 硬件计数器：`AiCoreNone` / `PipeUtilization` / `ArithmeticUtilization` / `Memory` 等 | Level0/1 用 None   |
| `l2_cache`            | 是否采集 L2 缓存命中率                                                                         | 深度分析时开启           |
| `op_attr`             | 是否记录算子属性                                                                              | 深度分析时开启           |
| `data_simplification` | 是否简化数据（丢弃部分细节）                                                                        | `False` 保留完整数据    |
| `record_op_args`      | 是否记录算子输入参数                                                                            | 深度分析时开启           |
| `gc_detect_threshold` | GC 检测阈值                                                                               | `None`            |

### 2.2 profile() 参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `activities` | 采集范围：CPU / NPU | 通常两者都开 |
| `schedule` | 采集调度策略 | 见下方说明 |
| `on_trace_ready` | 采集完成回调 | `tensorboard_trace_handler` |
| `record_shapes` | 记录 tensor shape | 需要时开启，有开销 |
| `profile_memory` | 记录内存分配/释放 | 需要分析显存时开启 |
| `with_stack` | 记录 Python 调用栈 | `True`，便于定位热点 |
| `with_modules` | 记录 nn.Module 层级 | `True`，便于定位模块 |
| `with_flops` | 记录浮点运算量 | 需要时开启 |

### 2.3 schedule() 参数

```
|-- skip_first --|-- wait --|-- warmup --|-- active --|  (× repeat)
```

| 参数 | 说明 |
|------|------|
| `skip_first` | 开始采集前跳过的 step 数 |
| `wait` | 每轮采集前等待的 step 数 |
| `warmup` | 预热 step 数（采集但不记录） |
| `active` | 实际记录的 step 数 |
| `repeat` | 重复采集轮数，0 表示不限 |

常用配置：
- 快速抓一个 step：`schedule(wait=0, warmup=0, active=1, repeat=0, skip_first=0)`
- 跳过前 10 步再抓 3 步：`schedule(wait=0, warmup=0, active=3, repeat=0, skip_first=10)`
- 预热后抓取：`schedule(wait=0, warmup=2, active=1, repeat=0, skip_first=5)`

## 三、代码中使用 profiler

```python
# 方式 1：context manager
with prof:
    for step, batch in enumerate(dataloader):
        # 训练逻辑
        loss = model(batch)
        loss.backward()
        optimizer.step()
        prof.step()  # 每个 step 结束时调用

# 方式 2：手动 start/stop
prof.start()
for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()
    prof.step()
prof.stop()
```

## 四、采集数据查看

### 4.1 采集输出结构

采集后在输出目录下会生成类似如下的目录（非 JSON 文件）：

```
<输出目录>/
└── kmaker-cz50c-010233160106_878_20260423071201266_ascend_pt/
    └── PROF_xxxx/
        ├── device_x/
        │   ├── summary/       # CSV 格式的算子统计
        │   └── timeline/      # timeline 数据
        └── host/
```

这是 CANN 原生的 **msprof 格式**，不是 Chrome Trace JSON。

### 4.2 方式 1：msprof 转换为可视化数据（推荐）

```bash
# 解析 profiling 数据，生成 timeline JSON 和汇总报告
msprof --export=on --output=<ascend_pt 目录的完整路径>

# 示例
msprof --export=on --output=/sfs_turbo/hw/profile/kmaker-cz50c-010233160106_878_20260423071201266_ascend_pt
```

解析后会在目录下生成：
- `timeline/*.json` — 可用 Chrome Tracing 打开
- `summary/*.csv` — 算子耗时汇总

### 4.3 方式 2：Chrome Tracing 查看 timeline

1. 先用 `msprof --export=on` 转换出 JSON 文件
2. 浏览器打开 `chrome://tracing`
3. 拖入 `timeline/*.json` 文件
4. 可以看到 CPU/NPU 算子的时间线、耗时、调用关系

### 4.4 方式 3：TensorBoard 查看

```bash
pip install tensorboard torch-tb-profiler
tensorboard --logdir <输出目录>
```

浏览器打开 `http://localhost:6006`，在 PyTorch Profiler 插件页查看：
- Overview：算子耗时排序
- Trace View：CPU/NPU 时间线
- Memory View：显存分配时间线（需 `profile_memory=True`）
- Module View：按 nn.Module 聚合耗时（需 `with_modules=True`）

> 注意：昇腾环境下 TensorBoard 可能无法直接读取 msprof 格式，需要先用 msprof 转换。

### 4.5 方式 4：MindStudio 图形化分析

华为官方 IDE MindStudio 可以直接导入整个 `*_ascend_pt` 目录：
- 图形化展示算子耗时、timeline、内存
- 支持 AI Core 硬件指标分析（需 Level2 采集）
- 适合深度性能调优

### 4.6 方式 5：命令行快速查看 CSV

```bash
# 查找 CSV 汇总文件
find <ascend_pt目录> -name "*.csv" | head -20

# 查看算子耗时排序
column -t -s, <csv文件路径> | less
```

## 五、采集级别选择建议

| 场景 | 推荐级别 | 关键参数 |
|------|----------|----------|
| 快速定位性能瓶颈 | Level0 | `profile_memory=False` |
| 分析算子细节+显存 | Level1 | `profile_memory=True, with_modules=True` |
| 深度硬件级调优 | Level2 | `aic_metrics=PipeUtilization, l2_cache=True` |

## 六、注意事项

1. **采集开销：** Level 越高开销越大，Level2 会显著影响训练速度，建议 `active` 设为 1-2 个 step
2. **磁盘空间：** profiling 数据可能很大（数百 MB 到数 GB），确保输出目录有足够空间
3. **多卡场景：** 每张卡会生成独立的 profiling 数据，目录名中包含 device ID
4. **数据清理：** 分析完成后及时清理 profiling 数据，避免占用大量磁盘
5. **skip_first：** 建议跳过前几个 step（如 `skip_first=5`），因为前几步通常包含 warmup 和编译开销，不能代表稳态性能
