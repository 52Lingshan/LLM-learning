错误1
```
[36m(RayEngine pid=624550)[0m Traceback (most recent call last):

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/engine_server.py", line 324, in train_batch

[36m(RayEngine pid=624550)[0m     stats = await loop.run_in_executor(None, self.engine.train_batch, binary_data)

[36m(RayEngine pid=624550)[0m             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/concurrent/futures/thread.py", line 58, in run

[36m(RayEngine pid=624550)[0m     result = self.fn(*self.args, **self.kwargs)

[36m(RayEngine pid=624550)[0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/backend/megatron_backend.py", line 1230, in train_batch

[36m(RayEngine pid=624550)[0m     outputs, stats = self.engine.train(mb_inputs, forward_step,

[36m(RayEngine pid=624550)[0m                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/backend/megatron_backend.py", line 647, in train

[36m(RayEngine pid=624550)[0m     outputs = train_step(

[36m(RayEngine pid=624550)[0m               ^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/third_party/megatron/megatron_0_11_0/megatron_helper.py", line 336, in train_step

[36m(RayEngine pid=624550)[0m     losses_reduced = forward_backward_func(

[36m(RayEngine pid=624550)[0m                      ^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 636, in forward_backward_no_pipelining

[36m(RayEngine pid=624550)[0m     output_tensor, num_tokens = forward_step(

[36m(RayEngine pid=624550)[0m                                 ^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 423, in forward_step

[36m(RayEngine pid=624550)[0m     output_tensor, loss_func = forward_step_func(data_iterator, model)

[36m(RayEngine pid=624550)[0m                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/rl_function/actor_function.py", line 1330, in forward_step

[36m(RayEngine pid=624550)[0m     output = mcore_model_forward_packed(

[36m(RayEngine pid=624550)[0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Asystem-HybridEngine/asystem_runtime/rl_function/actor_function.py", line 1020, in mcore_model_forward_packed

[36m(RayEngine pid=624550)[0m     output_orig = model(

[36m(RayEngine pid=624550)[0m                   ^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl

[36m(RayEngine pid=624550)[0m     return self._call_impl(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl

[36m(RayEngine pid=624550)[0m     return forward_call(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Megatron-LM/megatron/core/distributed/data_parallel_base.py", line 22, in forward

[36m(RayEngine pid=624550)[0m     return self.module(*inputs, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl

[36m(RayEngine pid=624550)[0m     return self._call_impl(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl

[36m(RayEngine pid=624550)[0m     return forward_call(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Megatron-LM/megatron/core/transformer/module.py", line 489, in forward

[36m(RayEngine pid=624550)[0m     outputs = self.module(*inputs, **kwargs)

[36m(RayEngine pid=624550)[0m               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl

[36m(RayEngine pid=624550)[0m     return self._call_impl(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/usr/local/python3.11.14/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl

[36m(RayEngine pid=624550)[0m     return forward_call(*args, **kwargs)

[36m(RayEngine pid=624550)[0m            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/sfs_turbo/hw/shiyang/code/kimi25vl_maijia/Megatron-Bridge/src/megatron/bridge/models/kimi_vl/modeling_kimi_k25_vl.py", line 196, in forward

[36m(RayEngine pid=624550)[0m     inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(

[36m(RayEngine pid=624550)[0m                                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

[36m(RayEngine pid=624550)[0m   File "/root/.cache/huggingface/modules/transformers_modules/kimi_hyphen_mini_majia/modeling_kimi_k25.py", line 1008, in _merge_input_ids_with_image_features

[36m(RayEngine pid=624550)[0m     raise ValueError(

[36m(RayEngine pid=624550)[0m ValueError: The input provided to the model are wrong. The number of image tokens is 0 while the number of image features given to the model is 144. This prevents correct indexing and breaks batch generation.
```
