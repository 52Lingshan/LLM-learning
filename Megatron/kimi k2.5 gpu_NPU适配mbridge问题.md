#### 一、安装问题
1.用pip安装的话，这些包都需要--no-build-isolation，重点是"causal-conv1d，mamba-ssm

no-build-isolation-package = [

    "transformer-engine",

    "transformer-engine-torch",

    "mamba-ssm",

    "causal-conv1d",

    "nv-grouped-gemm",

    "flash_mla",

    "flash-linear-attention",

]

2.uv安装相对来说问题少很多



#### 二、PR问题
##### generation_config
File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/megatron/bridge/models/conversion/auto_bridge.py", line 908, in to_megatron_provider provider: ModelProviderMixin = self._model_bridge.provider_bridge(provider_input) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/megatron/bridge/models/kimi_vl/kimi_k25_vl_bridge.py", line 69, in provider_bridge provider = KimiK25VLModelProvider( ^^^^^^^^^^^^^^^^^^^^^^^ TypeError: KimiK25VLModelProvider.__init__() got an unexpected keyword argument 'generation_config'

修复：

 KimiK25VLModelProvider  

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773192464978-2cfb9ef9-78e4-4f01-b1f1-f1900cdbd1b6.png)

 复 KimiK25VLBridge  

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773192559162-44bb2a30-f8a4-4d9f-b728-64677e14845e.png)



##### Transfomer版本（5.0以上）和kimi k2.5模型（4.56.2）动态加载版本不一致
[rank0]:   File "/root/.cache/huggingface/modules/transformers_modules/modeling_kimi_k25.py", line 67, in  [rank0]:     from .modeling_deepseek import DeepseekV3ForCausalLM [rank0]:   **File "/root/.cache/huggingface/modules/transformers_modules/modeling_deepseek.py", line 47, in  [rank0]:     from transformers.utils.import_utils import is_torch_fx_available [rank0]: ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils' **(/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/utils/import_utils.py). Did you mean: 'is_torch_available'? [rank0]:[W310 22:10:45.081783027 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see [https://pytorch.org/docs/stable/distributed.html#shutdown](https://pytorch.org/docs/stable/distributed.html#shutdown) (function operator())



Megatron-Bridge要求transformers版本是5.0以上，所以修改transformers代码

/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/utils/import_utils.py

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773192823799-0e922689-19d5-47c5-a582-48a8c6f0ff5a.png)



##### FA2版本问题
[rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 1928, in get_correct_attn_implementation  
[rank0]:     self._flash_attn_2_can_dispatch(is_init_check)  
[rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 1587, in _flash_attn_2_can_dispatch  
[rank0]:     raise ImportError(f"{preface} the package flash_attn seems to be not installed. {install_message}")  
[rank0]: ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package flash_attn seems to be not installed. Please refer to the documentation of [https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2) to install Flash Attention 2.  
[rank0]:[W311 09:21:17.719999838 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources.

通过uv pip install flash-attn --no-build-isolation --extra-index-url [https://pypi.antfin-inc.com/simple/](https://pypi.antfin-inc.com/simple/)

安装时候出现：nvme5n1p1: write failed, project block limit reached. nvme5n1p1: write failed, project block limit reached.以及编译报错



修复：关闭FA2, 修改config.json,修改_attn_implementation为eager

"vision_config": {  
    "_attn_implementation": "eager",  
    "init_pos_emb_height": 64,  
    "init_pos_emb_time": 4,  
    "init_pos_emb_width": 64,  
    "merge_kernel_size": [  
      2,  
      2  
    ],

##### 单机减层跑的时候，出现TypeError: KimiK25ForConditionalGeneration.tie_weights() got an unexpected keyword argument '<font style="color:#DF2A3F;">recompute_mapping</font>' 和 '<font style="color:#DF2A3F;">missing_keys</font>'
ise you'll get an exception).

+ If you are not the owner of the model architecture class, please contact the model code owner to update it.  
[rank0]: Traceback (most recent call last):  
[rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 1067, in  [rank0]:     compare_models_one_step(args) [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 862, in compare_models_one_step [rank0]:     hf_model = _load_hf_model(args, is_vl_model) [rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 575, in _load_hf_model [rank0]:     hf_model = model_class.from_pretrained( [rank0]:                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 367, in from_pretrained [rank0]:     return model_class.from_pretrained( [rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4094, in from_pretrained [rank0]:     model = cls(config, *model_args, **model_kwargs) [rank0]:             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [rank0]:   File "/root/.cache/huggingface/modules/transformers_modules/modeling_kimi_k25.py", line 851, in **init** [rank0]:     self.post_init() [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 1333, in post_init [rank0]:     self.init_weights() [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/megatron-bridge-venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 3090, in init_weights [rank0]:     self.tie_weights(recompute_mapping=False) [rank0]: TypeError: KimiK25ForConditionalGeneration.tie_weights() got an unexpected keyword argument 'recompute_mapping'

自定义的 KimiK25ForConditionalGeneration 类重写了 tie_weights() 方法，但未兼容 Transformers 新版本新增的 recompute_mapping 和missing_keys参数。

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773213394047-eb189c7f-348e-4690-b4c3-13e4a2f4a0e6.png)修复方案：

**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">def tie_weights(</font>****self****<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">,</font>**** missing_keys****<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">:list|None=None,</font>**** recompute_mapping****<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">:bool=False):</font>**



##### ValueError: Provide either 'messages' or both 'medias' and 'text'
Processing inputs - Prompt: 'Hello, how are you?', Image: None  
[rank0]: Traceback (most recent call last):  
[rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 1067, in  [rank0]:     compare_models_one_step(args) [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 869, in compare_models_one_step [rank0]:     input_ids, pixel_values, image_grid_thw, messages = process_inputs( [rank0]:                                                         ^^^^^^^^^^^^^^^ [rank0]:   File "/workspace/bin/wangliping/Megatron-Bridge/examples/conversion/compare_hf_and_megatron/compare.py", line 492, in process_inputs [rank0]:     inputs = processor(text=[prompt], return_tensors="pt") [rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [rank0]:   File "/root/.cache/huggingface/modules/transformers_modules/kimi_k25_processor.py", line 97, in **call** [rank0]:     raise ValueError( [rank0]: ValueError: Provide either 'messages' or both 'medias' and 'text'

 Kimi K25 处理器（processor）要求必须传入 `<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">messages</font>` 参数（多模态对话格式），或同时传入 `<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">medias</font>` 和 `<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">text</font>`，但你只传入了 `<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">text</font>`

<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">修改compare.py</font>

```plain
def process_inputs(tokenizer, processor, image_path: Optional[str], prompt: str, is_vl_model: bool, tp_size: int = 1):
    """Process inputs for both vision-language and regular LLM models."""
    if is_vl_model and image_path:
        if _is_kimi_processor(processor):
            # Kimi K2.5: use processor(messages=messages) directly
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image_url": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = processor(messages=messages)
            input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
            grid_thws = getattr(inputs, "grid_thws", None)
            return input_ids, inputs.pixel_values, grid_thws, messages
        elif QWEN_VL_UTILS_AVAILABLE and processor is not None:
            # Qwen VL and other models: use process_vision_info + processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
            return input_ids, inputs.pixel_values, inputs.image_grid_thw, messages
        else:
            # Processor unavailable -- generate synthetic vision inputs so we
            # can still exercise the vision forward path with random data.
            print_rank_0("Processor unavailable; generating synthetic vision inputs for testing.")
            return _generate_synthetic_vision_inputs(tokenizer, prompt, tp_size)
    else:
        # Text-only processing for both VL models without images and regular LLMs
        if is_vl_model and processor and _is_kimi_processor(processor):
            # 核心修复：Kimi K25 纯文本输入使用 messages 格式
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            inputs = processor(messages=messages, return_tensors="pt")
        elif is_vl_model and processor:
            # Use processor for other VL models even in text-only mode
            inputs = processor(text=[prompt], return_tensors="pt")
        else:
            # Use tokenizer for regular LLMs
            inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = pad_input_ids_to_tp_multiple(inputs.input_ids, tp_size, tokenizer.pad_token_id or 0)
        return input_ids, None, None, None
```



##### AttributeError: type object 'DynamicCache' has no attribute 'from_legacy_cache'
<font style="color:rgba(0, 0, 0, 0.85);background-color:rgba(0, 0, 0, 0.04);">Transformers 版本兼容性问题 —— DynamicCache.from_legacy_cache() 是较新版本 Transformers 新增的方法，但你的环境中 Deepseek 模型代码适配了新版本 API</font>

<font style="color:rgba(0, 0, 0, 0.85);background-color:rgba(0, 0, 0, 0.04);">修改config.json文件，关闭use_cache</font>

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773216215469-c1fec452-f8f4-4e89-8ae1-d5b971725805.png)

关闭后正常运行结果

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773216330052-5161149a-3260-4e0e-9d23-c89a6c64f1f4.png)

MindspeedTeLinear no p

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773808479933-8f4a886f-2fc8-40f7-8b41-f7e73664c0c6.png)

##### 权重加载expert bias shape 对不齐，<font style="color:rgba(0, 0, 0, 0.9);background-color:rgba(0, 0, 0, 0.03);">language_model.model.layers.1.mlp.gate.e_score_correction_bias</font>
<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773553198625-95a487b8-fc59-44ac-9774-73a653cd0624.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773557339438-e5a7265f-e520-4c30-83ff-68ab6ae6b870.png)

打印了下，确实这个路径是384的

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773557357849-6ccccac3-867b-4f25-a16c-c32fcf8ff576.png)

改成384 权重加载能过，但是跑纯megatron-bridge测试 得16，应该是hybridengine那边得适配下@WB02075302



##### class KimiK25VLModel(MegatronModule): 没适配packed_seq_params
<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773557307719-c0b07ddd-b59c-472c-a25c-ddfab56f1cf2.png)

/sfs_turbo/hw/jingyu_P/qwen35_train/0310/Megatron-Bridge/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py

/sfs_turbo/hw/jingyu_P/qwen35_train/0313/Megatron016/Megatron-Bridge/src/megatron/bridge/models/kimi_vl/modeling_kimi_k25_vl.py

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773573380611-8b1ef809-60eb-4a60-972f-99fd1aff1a6a.png)

内仓已修复，手动copy到本地，后边要合一下代码提交到内仓

[https://code.alipay.com/Theta/Megatron-Bridge/pull_requests/3](https://code.alipay.com/Theta/Megatron-Bridge/pull_requests/3)

moe_permute_fusion=False,没适配

ypeError: Module.__init__() takes 1 positional argument but 5 were given

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773414741424-cee71b44-1f0c-4677-a70d-c04ed12ae80d.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773566058196-3eeb7e32-07b8-4c14-b776-906e8126b146.png)

代码里写死后跑通@Mindseed适配

##### 权重保存is_quantized不存在，注掉跑通@Mindseed适配
<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/186556424/1773566125843-7bc15477-b735-4e6d-9aa3-7fbdc351a8b6.png)



10. cannot import name

[rank14] : ImportError:

    transformers.utils.import_u

tils' (/usr/local/lib/python3.11/site-packages/transformers/utils/import_utils.py)



<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773653178063-8f297bf5-1d67-4ed4-87a5-16c5fb1de8c0.png)

增加函数

```shell
@lru_cache
def is_fouroversix_available() -> bool:
    return _is_package_available("fouroversix")
```

#### 三、下一步计划
##### 3.1 华为研发正向解决兼容性问题


##### 3.2 基于Megatron-bridge + transformers 4.56.2验证兼容性问题
###### 3.2.1 ImportError: cannot import name 'Qwen3NextForCausalLM' from 'transformers' 
transformers版本改为**4.57.6**能兼容Megatron-bridge最新main分支代码，并且可以开启use_cache

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773284332853-98334e05-8a9e-43a4-aa5a-a0d95a3b53db.png)



###### 3.3 兼容性分析
1. 4.57.6和megatron-bridge 可能有兼容性问题 

优点：兼容改动少

缺点：镜像不统一

2. 5.2.0 和kimi2.5的相关模型文件有兼容性问题

(1) t兼容import_utils.py，增加两个函数

(2) 兼容KimiK25ForConditionalGeneration，适配tie_weights()

(3) DynamicCache兼容性，config关闭use_cache

优点：和qwen3.5能统一镜像，后续kimi升级后无需再升级

缺点：兼容性改动多，镜像不统一



#### 四、单机预训练实验
1.配置kimi25cofig

```plain
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _get_kimi_k2_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for Kimi-K2 based on PP and VP size."""
    map_pp_vp_to_layout = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (2, 8): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
    }

    vp_size = 1 if vp_size is None else vp_size
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for Kimi-K2. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def kimi_k25_pretrain_config(optimizer_type: str = "muon", hf_path: str = "/dnn_training_sys/users/shiyang/models/Kimi-K2.5_bf16/") -> ConfigContainer:
    """Return a pre-training config for Kimi-K2 (1T).

    Recommended parallelism: TP=2, PP=16, EP=32

    Args:
        optimizer_type: 'adam' or 'muon' (default).
    """
    cfg = _pretrain_common()

    # Model config - uses KimiK2Provider instead of AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)

    # Tokenizer

    cfg.tokenizer.tokenizer_model = "/dnn_training_sys/users/shiyang/models/Kimi-K2.5_bf16/"
    cfg.tokenizer.trust_remote_code = True
    cfg.model.seq_length = 4096

    cfg.model.num_workers = 8
    cfg.logger.log_interval = 1
    cfg.model.use_flash_attn = True

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    # Set pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_kimi_k2_pipeline_layout(16, 1)

    # Pipeline split settings (asymmetric stages)
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Dataset config - mock data by default
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer
    if optimizer_type == "adam":
        opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=2500,
            lr_decay_iters=cfg.train.train_iters,
            max_lr=3e-4,
            min_lr=3e-5,
        )
    elif optimizer_type == "muon":
        opt_cfg, scheduler_cfg = distributed_muon_with_cosine_annealing(
            lr_warmup_iters=2000,
            lr_decay_iters=cfg.train.train_iters,
            max_lr=3e-4,
            min_lr=3e-5,
        )
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading) - already set in KimiK2Provider
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - Kimi-K2 uses custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    # Adam uses grad_reduce_in_fp32=False, Muon uses True.
    grad_reduce_in_fp32_default = optimizer_type != "adam"
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=grad_reduce_in_fp32_default,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False  # Pad router for FP8 alignment

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = True

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config — Adam uses distributed optimizer + overlap; Muon requires both off.
    if optimizer_type == "adam":
        cfg.ddp.use_distributed_optimizer = True
        cfg.ddp.overlap_param_gather = True
        cfg.ddp.grad_reduce_in_fp32 = False
    else:
        cfg.ddp.use_distributed_optimizer = False  # Muon needs this to be False
        cfg.ddp.overlap_param_gather = False  # Muon needs this to be False
        cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg

```

2.设置脚本

/sfs_turbo/hw/jingyu_P/qwen35_train/0313/Megatron016/Megatron-Bridge/examples/decentralized_pg/pretrain_kimi25.py

```plain
#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     Apache License, Version 2.0 | Apache Software Foundation
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3.5-VL 35B-A3B Pretraining Script

This script provides a direct Python interface to run Qwen3.5-VL 35B-A3B pretraining
without using SLURM. It's useful for local testing or environments without SLURM.

Usage:
    python pretrain_qwen35_vl_35b_a3b.py --help

    # Run with default settings (mock data)
    python pretrain_qwen35_vl_35b_a3b.py

    # Run with custom settings
    python pretrain_qwen35_vl_35b_a3b.py \
        --seq-length 4096 \
        --train-iters 100 \
        --global-batch-size 32 \
        --micro-batch-size 1 \
        --tp 2 \
        --pp 1 \
        --ep 16

Note:
    This script uses the VLM-specific forward_step function from vlm_step.py,
    which handles both text and visual inputs for multimodal pretraining.
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from megatron.bridge.recipes.kimi import kimi_k25_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step as vlm_forward_step


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Qwen3.5-VL pretraining."""
    parser = argparse.ArgumentParser(
        description="kimi25 pretrain Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model and data settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen35_vl_35b_a3b",
        help="Model name (default: qwen35_vl_35b_a3b)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length for training (default: 4096)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="mock",
        choices=["mock"],
        help="Dataset type (default: mock)",
    )

    # Parallelism settings
    parser.add_argument(
        "--tp",
        type=int,
        default=2,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Pipeline parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=16,
        help="Expert parallelism degree (default: 16)",
    )
    parser.add_argument(
        "--cp",
        type=int,
        default=1,
        help="Context parallelism degree (default: 1)",
    )
    parser.add_argument(
        "--sp",
        type=bool,
        default=True,
        help="Sequence parallelism (default: True)",
    )

    # Training settings
    parser.add_argument(
        "--train-iters",
        type=int,
        default=300000,
        help="Number of training iterations (default: 300000)",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=32,
        help="Global batch size (default: 32)",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Maximum learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=3e-5,
        help="Minimum learning rate (default: 3e-5)",
    )
    parser.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=500,
        help="Learning rate warmup iterations (default: 500)",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="Checkpoint saving interval (default: 500)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory (default: ./checkpoints)",
    )

    return parser.parse_args()


def main() -> None:
    """Main function for Qwen3.5-VL 35B-A3B pretraining."""
    args = parse_args()

    logger.info(f"Starting Qwen3.5-VL 35B-A3B pretraining with args: {args}")

    # Load default configuration
    cfg: ConfigContainer = kimi_k25_pretrain_config()

    # Override configuration with command-line arguments
    overrides = {
        "model.tensor_model_parallel_size": args.tp,
        "model.pipeline_model_parallel_size": args.pp,
        "model.expert_model_parallel_size": args.ep,
        "model.context_parallel_size": args.cp,
        "model.sequence_parallel": args.sp,
        "train.train_iters": args.train_iters,
        "train.global_batch_size": args.global_batch_size,
        "train.micro_batch_size": args.micro_batch_size,
        "scheduler.max_lr": args.lr,
        "scheduler.min_lr": args.min_lr,
        "scheduler.lr_warmup_iters": args.lr_warmup_iters,
        "logger.log_interval": args.log_interval,
        "checkpoint.save_interval": args.save_interval,
        "checkpoint.save": args.save_dir,
        "checkpoint.load": args.save_dir,
    }

    # Process configuration with overrides
    # cfg = process_config_with_overrides(cfg, overrides)

    # Run pretraining
    print("=========cfg.model.num_layers:{cfg.model.num_layers}==========", cfg.model.num_layers)
    pretrain(cfg, vlm_forward_step)


if __name__ == "__main__":
    main()
```

3.需要修改tokenizer.py

kwargs["trust_remote_code"] = True

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773387993173-ccfffbad-f5d7-43d6-9455-cbc1356d2b24.png)



4.单机单卡运行

1.python examples/decentralized_pg/pretrain_kimi25.py

2.单机多卡运行 python -m torch.distributed.run --nproc_per_node=16 examples/decentralized_pg/pretrain_kimi25.py



#### 五、Megatron-Birdge 训练GPU与NPU精度对比
##### 1、GPU精度基线采集
###### 1、GPU环境
###### [https://pnantshellweb.antgroup-inc.cn/terminal/online?container=head&hostname=aistudio-arnjpekt-rayjob&key=hostname&source=dashboard&tenant=tmc-public-su18-hn](https://pnantshellweb.antgroup-inc.cn/terminal/online?container=head&hostname=aistudio-arnjpekt-rayjob&key=hostname&source=dashboard&tenant=tmc-public-su18-hn)
###### 2、命令
（1）source /workspace/bin/megatron-env_2743/bin/activate

（2）cd /workspace/bin/Megatron-Bridge_2743/Megatron-Bridge

（3）export PYTHONPATH=/workspace/bin/SGLang/python:/workspace/bin/flash-linear-attention:/workspace/bin/Asystem-HybridEngine/ainsight/lib:/workspace/bin/amed_areal_RL_reward:/workspace/bin/Asystem-HybridEngine:/workspace/bin/AReaL:/workspace/bin/AReaL/standalone_megatron:/workspace/bin/antllm:/workspace/bin/atorch:/opt/conda/lib/python3.8/site-packages/aistudio_notebook/public::/root/workdir/astra-build/Asystem-HybridEngine/astra_cache/astra-client/python



（4）torchrun --nnodes=1 --nproc_per_node=8 pretrain.py

###### 3、训练关键配置
(1) 并行、切分参数设置

/workspace/bin/Megatron-Bridge_2743/Megatron-Bridge/src/megatron/bridge/recipes/kimi/kimi_k25.py

##### 2、NPU精度数据采集
###### 命令
1.cd /sfs_turbo/hw/jingyu_P/qwen35_train/0313/Megatron016/Megatron-Bridge

2.source /sfs_turbo/hw/wangliping/kimi25/set_env.sh

3.执行python -m torch.distributed.run --nproc_per_node=8 examples/decentralized_pg/pretrain_kimi25.py

<font style="color:#DF2A3F;">注意</font>

每次跑完训练，删除生成的checkpoint，不然就是断点续训了，save路径/sfs_turbo/hw/jingyu_P/qwen35_train/0313/Megatron016/Megatron-Bridge/nemo_experiments里面的checkpoints文件夹



配置文件参数修改位置：

/sfs_turbo/hw/jingyu_P/qwen35_train/0313/Megatron016/Megatron-Bridge/src/megatron/bridge/recipes/kimi/kimi_k25.py

```plain
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def _get_kimi_k2_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for Kimi-K2 based on PP and VP size."""
    map_pp_vp_to_layout = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + ["loss"]],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
        (2, 8): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder", "loss"]],
    }

    vp_size = 1 if vp_size is None else vp_size
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for Kimi-K2. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def kimi_k25_pretrain_config(optimizer_type: str = "adam", hf_path: str = "/sfs_turbo/pretrained_models/Kimi-K2.5-bf16") -> ConfigContainer:
    """Return a pre-training config for Kimi-K2 (1T).

    Recommended parallelism: TP=2, PP=16, EP=32

    Args:
        optimizer_type: 'adam' or 'muon' (default).
    """
    cfg = _pretrain_common()

    # Model config - uses KimiK2Provider instead of AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)

    # Tokenizer

    cfg.tokenizer.tokenizer_model = "/sfs_turbo/pretrained_models/Kimi-K2.5-bf16"
    cfg.model.seq_length = 4096
    # cfg.model.deterministic_mode = True
    cfg.model.num_workers = 16
    cfg.logger.log_interval = 1
    cfg.model.use_flash_attn = True

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 3
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Optimizer - higher LR for pretraining
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.pack_sequences_in_batch = False

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.ckpt_cfg.ckpt_format = 'torch'
    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"

    # Test.
    # cfg.checkpoint.load = "/home/y30067824/bridge_train/yzy/Megatron016/Megatron-Bridge/src/nemo_experiments/default/checkpoints/"
    # cfg.checkpoint.load_main_params_from_ckpt = True
    # cfg.checkpoint.finetune = True

    return cfg
```



##### 3、kimi k2.5 Megatron bridge精度问题定位
######  问题现象<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/197256595/1773482084986-e238c4a1-ddbb-4488-ba1f-7680ff3f874c.png)
gpu两次曲线一致

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/197256595/1773482144022-f0134035-bafc-4c9a-b68a-5c29c798e88b.png)

npu两次曲线一致

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/197256595/1773482184224-19da8d80-b406-4a91-b52f-283fa7cab13d.png)

GPU与NPU曲线不一致

###### 问题根因
原因是因为加载权重时候用的是随机权重，不是真正加载权重，修改代码

**<font style="color:#DF2A3F;">load_weights=True</font>**

```python
# Model config - uses KimiK2Provider instead of AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=True)
```



###### 加载权重后，误差减少，但是问题仍存在


<!-- 这是一张图片，ocr 内容为：GRAD NORM CHART LOSS CHART 0G 导员品 10 5 0 WAYMARAMMUN 0 -5- -10 6080100120 140160180 20 100 120 160 20 140 0 0 40 NPU_16_TRAIN2.LOG NPU_16_TRAIN2.LOG CANVASJS.COM COMPARISON ABSOLUTE CHART COMPARISON ABSOLUTE CHART 0.06 0.04 0.02 REYYNIN WINTHU -2 -0.02 0 100 20 20 100 60 40 140 80 80 120 60 160 140 160 180 120 ERROR ERROR CANVASJS.COM CANVASJS.COM MEAN ERROR:1.09592999999999998,MEAN SQUARE ERROR:2.1347671100001 MEAN ERROR:0.009725458250000022,MEAN SQUARE ERTOR:0.000237506994560056 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/151756451/1773662392799-f8042ccc-9cb9-4b7c-ac72-9eaa57e66c35.png)



dump数据后初步判断为expert 反向精度问题

<!-- 这是一张图片，ocr 内容为：E D B H C A T SI ENCH NPU DTY ENCH DT PU TENSOR NPU NAME MAX  DIF MIN DILY REGUIX CH TENSOR REQUIRE MODULE.0.TORCH. FLETORCH. FLE[1,4096] MODULC.0.MODULC.FLOATI6NODULC.FORWARD.0.OUTPUTPUT.0 -0.0007705E-0.0046280.0 [1.1096] TRUO TRUE MODULE.0.TORCH.FLCTORCH.FL([1,4096] MODULE.0.MODULE.MODULE.KINIK25VLMODEL.FORWARD.0.OUTPUT.0 -0.00077056-0.0046280. [1,4096] TRUE TRUE HODULE.0.MODULE.NODULE.LANSUASE MODEL.DECODER.LAYERS,LAYERS,L.ALP.EXPERTS.TEGROUPEDHLP.BACKRARD.O MODULE.0.TORCH.BFL TORCH.BFI[12246,7168] -3.051757810 [12256,7168]FALSE FALSE 9.5367431630 LANSUPER,  MOFULE,NODULA, LANSUAFO,MODEI. DECOFER, LATERS, L.ALP, DZPERES, TBCROUPEANLP.BUCKSARD.OUT. MODULE.0.TORCH.BFLTORCH.BF][12246,7168] FALSE 7168JFALSE [12256, 1,474982126-2.9806833. H MODUL8,0,MADALE,NODELE,LANGUARB MODEL, DESADER, LARERS,L.ELB,ELB, GAPERES,TEGROUPEREAREARB.B.DUTPUT MODULE.0.TORCH.FLCTORCH.FL FALSE MODULE, NODULE, DECODULE, LANGUARB,MODEL, DECODER, IAYERS, L.ALP,EXPERTS, TEGROUPEDULP. FORD. INPUT. -0.001953120 TRUE MODULE.0.TORCH. BF.TORCH.BF [12246, 7168] 5,4836273195088034 MODULC, MODULC,   NODULE,   IANGUARE MODUL  DECODER,  LAYORS,  INPUT,EXPERTS, TEGROUPEDMLP,O,INPUT,3 MODULE. 0.TORCH.FLETORCH. FL TRUE [12246] 12256 0 HODULE.0,NODULE,NODULC,LANSUARE MODEL, DECODER.LAYORS, LAYORS, LANSUTPUT. TEGROUPEDILP, 0, O,OUTPUT. MODULE.0.TORCH.BFLTORCH.BF][12246. 5, 7168 TRUE [12256, 7168 CUE -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/151756451/1773740936997-de3c3137-e334-4324-9e52-2d8f30f7ca68.png)

怀疑路由不一致导致的shape不一致



<!-- 这是一张图片，ocr 内容为：FX BENCH NAME X AA AB Z RE AC H NPU ME IPU 12N BENCH N BENCH N I INCH ML `INCH 12 HRELATI ARELATI GRADVRESULTR MAX DI MIN DI MEAN DI ZNORM D RELATI NPU MI STACK NPU ME REQUI MOSS HFO 16.56351 7.242672 57895 812 1331 16. 16. 274246 12.  12. 57864 812 1116 34048000244110.002653(16. -0.000481-0.03157:00030710.021545:002901'0.4340480 TRUE VARNING RESULT 00030710.021545-0029010043404870. 6 TRUE 3.16.56351 7.242672 12 57895 812.1331 16 56399 7.274246 12 57864 812 1116 00244110.002653016.56 0.000481-0.0318710 WARNING 排序 2.384185'0 1.818989-3.05175'0.60240900 34  TRUE 1. 273886-0.497513.988-05/-5.28-05 1.158-10 0006103)3.96E-05/-5. 2E-05/1,                    2738 FALSO WARNING 升序 +降序 按颜色: 筛选器 无 按颜色: 等于 WARNING 与 或者 0 选择一个 能量 (全选) PASS WARNING 自动应用 清除铸选器 应用筛选器 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/151756451/1773742195813-5bf9994d-1ffb-4dfa-a5ac-7d9a6fe43c37.png)

固定路由后，dump数据比对不存在error，warning继续分析



固定路由之后，loss可以对齐，gradnorm前面几步还是有一些区别

<!-- 这是一张图片，ocr 内容为：LOSS CHART GRAD NORM CHART 15 50 40 10 3000 5 0 0 -5 10 O O 50 50 150 150 200 200 100 250 250 100 -NPU_FIX_ROUTER_0317_2.LOG NPU_FIX ROUTER_0317_2.LOG CANVASJS.COM CANVASJS.COM COMPARISON ABSOLUTE CHART COMPARISON ABSOLUTE CHART 0.015 0.01 0.005 MANUVYMA WOMMINARRUMY MEMEMEME MIMUMM -0.005 50 50 100 200 100 250 150 250 0 200 150 0 -ERROR ERROR CANVASJS.COM CANVASJS.COM MEAN ERROR:0,51752999999999996. MEAN SQUARE ERTOR:0.823176783333337 MEAN ERROR:0.00070644959999999852,MEAN SQUARE ERROR:000025199257994918884 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/151756451/1773740877738-473f1fa4-210c-46d2-927f-a433392b2d82.png)



固定路由，TP=1的情况下，loss 和grad norm可以对齐

<!-- 这是一张图片，ocr 内容为：LOSS CHART GRAD NORM CHART 15 60 10 40 5 20 0- 5- 2030405060 60708090 10203040 70 60 50 90 80 10 -GPU_TP1_0318.LOG-NPU_TP1_0318.LOG -GPU_TP1-0318.LOG NPU_TP1-0318.LOG CANVASJS.COM COMPARISON ABSOLUTE CHART COMPARISON ABSOLUTE CHART 0.001 0.0008 0.0006 0.0004 0.5 0.0002 10 20 50 30 70 80 60 70 30 60 50 90 40 20 10 80 ERROR -ERROR CANVASJS.COM CANVASJS.COM MEAN ERROR:0.00015745770000006153.MEAN SQUARE ERROR:5.882961646302095E-8 MEAN ERROR:0.082190000000000014, MEAN SQUARE ERROR:0.03381231 -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/151756451/1774232795474-cff3e56d-a4e8-4da9-9100-57eda52ec415.png)

#### 六、Megatron-Birdge 训练GPU与NPU精度对比
##### 1.精度工具安装
pip install mindstudio-probe --pre

安装成功

Successfully installed mindstudio-probe-{version}

<font style="color:rgb(37, 43, 58);background-color:rgb(246, 247, 249);"></font>

##### <font style="color:rgb(37, 43, 58);background-color:rgb(246, 247, 249);">2.相关配置</font>
<font style="color:rgb(37, 43, 58);background-color:rgb(246, 247, 249);">详细配置参考</font>[https://gitcode.com/Ascend/msprobe/blob/master/README.md](https://gitcode.com/Ascend/msprobe/blob/master/README.md)

```shell
#增加初始化
from msprobe.pytorch import seed_all, PrecisionDebugger
seed_all()
debugger = PrecisionDebugger(config_path="/sfs_turbo/hw/wangliping/kimi25/config.json")

#增加
    # TRAINING
    if not config.validation.skip_train:
        if state.train_state.do_train and config.train.train_iters > 0:
            debugger.start(model)  #开始
            train(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                state,
                ckpt_context,
                pg_collection,
                callback_manager=callback_manager,
            )
            debugger.stop() #结束
            debugger.step() #采集
        barrier_and_log("after training is done")

```

```shell
{
    "task": "statistics",
    "dump_path": "/sfs_turbo/hw/wangliping/kimi25/npu-m-bridge-dump",
    "rank": [0],
    "step": [],
    "level": "L0",
    "async_dump": false,
    "extra_info": true,

    "statistics": {
        "scope": [], 
        "list": [],
        "tensor_list": [],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    }
}
```

2.1GPU精度dump

配置好之后，迭代设置1，重新跑一次pretrain

2.2NPU精度dump

配置好之后，迭代设置1，重新跑一次pretrain



##### 3.比对
具体文档[https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md#%E5%91%BD%E4%BB%A4%E6%A0%BC%E5%BC%8F](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md#%E5%91%BD%E4%BB%A4%E6%A0%BC%E5%BC%8F)

msprobe compare -tp /sfs_turbo/hw/chenzeng/kimi_k25/Megatron016/Megatron-Bridg_JY/dump-npu -gp /sfs_turbo/hw/chenzeng/kimi_k25/Megatron016/Megatron-Bridg_JY/dum-gpu  -o ./output_dump

[compare_result_20260316193115.xlsx](https://yuque.antfin.com/attachments/lark/0/2026/xlsx/195056343/1773661056756-4254fcd1-bead-4598-8b80-63c6be02ca4d.xlsx)

有了dump数据之后分析，分析方案如下

[https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md#%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9%E7%BB%93%E6%9E%9C%E5%88%86%E6%9E%90](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md#%E7%B2%BE%E5%BA%A6%E6%AF%94%E5%AF%B9%E7%BB%93%E6%9E%9C%E5%88%86%E6%9E%90)

#### <font style="color:rgb(37, 43, 58);">比对结果（Result）</font>
<font style="color:rgb(37, 43, 58);">比对结果分为三类：pass、warning和error，优先级error > warning > pass</font>

<!-- 这是一张图片，ocr 内容为： -->
![](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/195056343/1773661159226-0ef53f06-59ec-425f-929d-475ad8182c2e.png)

<font style="color:rgb(37, 43, 58);"></font>

<font style="color:rgb(37, 43, 58);"></font>

