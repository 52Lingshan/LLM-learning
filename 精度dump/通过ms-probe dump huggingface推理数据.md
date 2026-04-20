安装msprobe
参考https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/msprobe_install_guide.md
pip install mindstudio-probe --pre

config配置
```json
{
  "task": "statistics",
  "dump_path": "/sfs_turbo/cls/ling2_5/vllm_dump",
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

```Python
from transformers import AutoModelForCausalLM, AutoTokenizer
from msprobe.pytorch import PrecisionDebugger

debugger = PrecisionDebugger(config_path='./config.json')

model_name = "/sfs_turbo/hw/xmd/ling-mini-2.5-256k-debug"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "system", "content": "You are Ling, an assistant created by inclusionAI"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(model.device)
debugger.start(model=model)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1,
    use_cache=False
)
debugger.stop()
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```