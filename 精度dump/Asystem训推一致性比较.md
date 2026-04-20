## 背景

Ling2.5-mini GRPO，训练到30step之后reward下降，发现起始logp diff较大（约2.2），相比GPU logp diff的0.009，相差过大，怀疑训推一致性存在问题。

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769066609513-e96d0b57-b3ce-4cd0-ba53-64e3d9aaba43.png)

准备dump训练前向和推理进行比较

## dump流程

pip install mindstudio-probe

### 数据构造

两边构造相同的 input_ids = torch.arange(128)

### 训练dump

HybridEngine分支：a3-dev/v1.3.0-test

训练框架：megatron+mindspeed。

因为训练侧也只采forward数据，所以asystem里直接复用compute_logp的代码。

1. 插入msprobe代码

asystem_runtime.backend.megatron_backend.MegatronBackend:forward

```Python
# 导入工具的数据采集接口
        from msprobe.pytorch import PrecisionDebugger, seed_all
        # 在模型训练开始前固定随机性
        seed_all()
        # 在模型训练开始前实例化PrecisionDebugger
        debugger = PrecisionDebugger("/storage/hw/yzr/dump/config.json")
        debugger.start(model=self.model_engine)
        with torch.no_grad():
            forward_backward_func = get_forward_backward_func()
            outputs = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=list(itertools.tee(iter(data), len(self.model_engine))) if self.megatron_config.dist_info.vpp_parallel else iter(data),
                model=self.model_engine,
                num_microbatches=num_microbatches,
                seq_length=max_total_len,
                micro_batch_size=self.micro_batch_size,
                forward_only=True
            )
            # 关闭数据 dump
            debugger.stop()
            debugger.step()  # 结束一个step的dump
```

dump配置

```json
{
    "task": "statistics",
    "dump_path": "/storage/hw/yzr/dump",
    "rank": [0],
    "step": [0],
    "level": "mix",
    "async_dump": false,

    "statistics": {
        "scope": [],
        "list": [],
        "data_mode": ["forward"],
        "tensor_list": [],
        "summary_mode": "statistics"
    }
}
```

采集数据用于后续比较

### 推理dump

1. vllm-ascend插桩msprobe代码，[feat: configurable dump · AntCode](https://code.alipay.com/Theta/vllm-ascend/commit/97bc217bf3921606a974c7eeaded6d58ae04e5ab)
2. 模型启动参数添加dump相关配置

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769066915970-ea8e7840-553d-472a-b1d9-1de839de76b5.png)

```json
{
  "task": "statistics",
  "dump_path": "/sfs_turbo/cls/ling2_5/vllm_dump_tp1ep1",
  "rank": [],
  "step": [],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

### 比较dump结果

比较dump可以使用分级可视化工具，工具安装过程可以参考

[AtomGit | GitCode - 全球开发者的开源社区,开源代码托管平台](https://gitcode.com/Ascend/mstt/tree/master/plugins/tensorboard-plugins/tb_graph_ascend)

1. 比较dump结果，生成vis文件

```
msprobe -f pytorch graph -i compare.json -o compare_result
```

`npu_path`使用训练dump结果，`bench_path`使用标杆dump结果，这里使用vllm的dump结果。

```
{
  "npu_path": "/mnt/sfs_turbo/hw/yzr/dump/step0/rank0",
  "bench_path": "/sfs_turbo/cls/ling2_5/vllm_dump_tp1ep1/step0/rank0",
  "is_print_compare_log": true
}
```

2. 启动tensorboard分级可视化服务

```
tensorboard --logdir compare_result/ --bind_all --port 60004
```

## 比较过程

1. 比较第一层输入输出，输入相同，输出统计值差异较大

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769067419362-30ab9b9c-97c6-4d03-81dc-ff63f62d777c.png)

2. 继续比较第一层内部，qkv，q_layernorm, k_layernorm输入输出均完全对齐，g_proj输入输出完全对齐。但是g_norm输入（即gla算子的输出）已经有较大误差

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769067624152-22b060e2-128d-4abb-9441-9d58baea4c82.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769068336141-1b7a0d10-d64a-4d71-a0e6-39308fdfa22e.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/137756802/1769068252063-16b02bc8-ed22-43d3-a985-c098814b17ca.png)

由于目前工具无法采集triton算子输入输出，无法直接比较gla算子输入输出。并且训练和推理使用了不同的gla算子，推理使用fused_recurrent，训练使用chunk_gla，但是排查gla算子的输入发现，训练侧没有rope相关操作，这会导致gla算子输入就有较大差异，导致gla算子输出不同。

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/jpeg/137756802/1769068236712-bf2e7c89-f89a-4e11-8f92-868aa771376e.jpeg)

## 单算子对比

单算子输入前打桩，算子结果保存下来对比。

```
query = torch.load(f"/sfs_turbo/cls/ling2_5/mini/q_{layer_idx}.pt")
key = torch.load(f"/sfs_turbo/cls/ling2_5/mini/k_{layer_idx}.pt")
value = torch.load(f"/sfs_turbo/cls/ling2_5/mini/v_{layer_idx}.pt")
self.slope = torch.load(f"/sfs_turbo/cls/ling2_5/mini/g_{layer_idx}.pt")

core_attn_out, _ = self.core_attention(
    query,
    key,
    value,
    g_gamma=self.slope,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=cu_seqlens_q, # for varlen training
    head_first=False,
)
torch.npu.synchronize()
torch.save(core_attn_out, "o1.pt")
```

## 结论

训练侧，LinearAttention需要给qk补充rope计算

## 后续对比

目前只对比了第一层的attention部分，后续对比可能存在问题：

1. 训推并行策略难以对齐：训练并行 tp1ep2(训练其实不支持该并行方式，代码后面会报错)。报错原因待排查。
2. triton算子当前工具无法采集

## profiling文件：

训练profiling： /storage/yzr/profiler/0129/

推理profiling：/sfs_turbo/cls/ling2_5/mini/profile_mix/kmaker-cz50c-010238165003_2009485_20260129121224895_ascend_pt

## 记录：

fla的结果输出：min值有差异

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769236321584-d5a1648d-ac05-42fd-a97a-e0ab16fa0d61.png)

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769236891119-8729a7ad-44c8-4c86-8f64-8d2f1b3e2c46.png)

mlp(dense):

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769253940656-c0839fe2-8084-4011-9b68-9bb9086c0186.png)

### Moe层差异对比

第2层moe结果差异：

**MaxRelativeErr：**0.5435%

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769416796656-02f7046b-487b-4dd1-9df7-8447399c2b11.png)

第3层moe差异：

**MeanRelativeErr：**1.2048%

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769416940814-e312b858-1f59-403a-bfb3-7eca2e12fc6d.png)

第4层moe差异：

**MinRelativeErr: 0.6098%**

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1769417005283-63d9c00c-bb9e-4332-9402-60f2f0765708.png)

rope开融合算子

"use_fused_rotary_pos_emb": True

### 对比列表：

cp=1,tp=1 减层到五层对比

|   |   |   |   |
|---|---|---|---|
|算子|修改点||进展|
|rope（FLA）|训练开启融合算子(五层下降0.1%)||完成|
|chunk_simple_gla|g_gamma推理改成bf16|4096序列在cann8.3上精度有差异；8.5上正常。|完成|
|pre_gate_norm(group rms norm)|无差异||完成|
|residual+out|无差异||完成|
|mlp层(dense)|无差异||完成|
|Moe sharedExperts层|无差异||完成|
|MoeGatingTopK|推理已经替换成训练的小算子，推理输入bf16修改成fp32（五层无影响；专家选择对齐）||完成|
|AddRmsNorm|不涉及||完成|
|MoeTokenUnpermute|||待排查|
|group matmul|训推同一个算子||待排查|
|FusedInferAttentionScore|||推理decode使用，暂时无法对比|
|swiglu|无差异||完成|
|FA(训练) /推理（ringMLA）|||有差异,待排查|
|self.hidden_size_per_attention_head = 192|五层下降0.4%||完成|
|||||

结果：

![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/193056718/1770295029709-cd478b0d-d0f0-4363-9a77-73902e5942e6.png)

## 其他

```
# 加载数据
x = torch.load('/sfs_turbo/hw/yzr/ling25/data128.pt', weights_only=False,map_location="cpu")
seq_sample = x["inputs"]

leng = 4096
# 1. 修改 seqlens - 将所有值改为 [[128]]
seq_sample.seqlens = {
    'ppo_loss_mask': [[leng]],
    'packed_input_ids': [[leng]],
    'old_logp': [[leng]],
    'advantages': [[leng]],
    'kl_rewards': [[leng]]
}

# 2. 修改 data 中的张量为长度为 128 的张量
# 对于 packed_input_ids，创建从 0 到 127 的张量
seq_sample.data['packed_input_ids'] = torch.arange(leng, dtype=torch.int64)

# 对于其他张量，创建长度为 128 的零张量（或适当的值）
seq_sample.data['ppo_loss_mask'] = torch.zeros(leng, dtype=torch.bool)
seq_sample.data['old_logp'] = torch.zeros(leng, dtype=torch.float32)
seq_sample.data['advantages'] = torch.zeros(leng, dtype=torch.float32)
seq_sample.data['kl_rewards'] = torch.zeros(leng, dtype=torch.float32)

# 如果需要，你可以修改 ppo_loss_mask 的一些值为 True（例如）
# seq_sample.data['ppo_loss_mask'][:10] = True

# 检查修改结果
print("修改后的 seqlens:", seq_sample.seqlens)
print("\n修改后的 data 张量形状:")
for key, value in seq_sample.data.items():
    print(f"{key}: {value.shape}")
print(f"\npacked_input_ids 的值: {seq_sample.data['packed_input_ids']}")

# 如果需要保存修改后的数据
torch.save(x, "/sfs_turbo/hw/yzr/ling25/data4096.pt")
```