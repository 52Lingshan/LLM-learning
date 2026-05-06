```yaml
    1 # Kimi-K2.5-mini 16B MoE 双机显存分析配置
    2 # 硬件: 2 节点 × 16 NPU = 32 NPU, 每卡 64GB HBM
    3 # 并行: TP=2, PP=4, EP=2, CP=2 → 2×4×2×2 = 32 ranks
    4 # 目标: 采集内存快照, 对比 K2.5 1T 模型的显存模式
    5
    6 data_loader:
    7   mode: online
    8   online_data_dir: '/mnt/sfs_turbo/yzr/gyy_Asystem/rollouts'
    9   seq_num_per_step: 32
   10   origin_seq_len: 1024
   11   mock_seq_len: 1024
   12   max_tokens_per_mb: 1024
   13
   14 megatron:
   15   # === 模型架构 (Kimi-K2.5-mini, 27层 MoE) ===
   16   num_layers: 27
   17   num_hidden_layers: 27
   18   hidden_size: 2048
   19   num_attention_heads: 64
   20   num_query_groups: 16
   21   qk_head_dim: 192
   22   v_head_dim: 128
   23   num_experts: 64
   24   expert-ffn-hidden-size: 2048
   25   moe_router_topk: 6
   26   padded_vocab_size: 1000000
   27
   28   # === 并行策略 (类似 K2.5 1T: 四维全部激活) ===
   29   tensor_model_parallel_size: 2       # TP=2
   30   pipeline_model_parallel_size: 4     # PP=4, 类似 K2.5 1T 的多 stage 模式
   31   expert_model_parallel_size: 2       # EP=2 (64/2=32 experts/rank)
   32   expert_tensor_parallel_size: 1
   33   context_parallel_size: 2            # CP=2, 触发 Ring CP 缓冲分配
   34   # DP = 32 / (2×4×2×2) = 1, 与 K2.5 1T 一致
   35
   36   # === PP 层数分配 (27层 → 4 stage) ===
   37   # first=4层, mid1=10层, mid2=10层, last=3层
   38   # 与 K2.5 1T 的不均匀分配类似, 便于观察 stage 间显存差异
   39   decoder_first_pipeline_num_layers: 4
   40   decoder_last_pipeline_num_layers: 3
   41   num_layers_in_first_pipeline_stage: 4
   42   num_layers_in_last_pipeline_stage: 3
   43
   44   # === 序列配置 ===
   45   seq_length: 1024
   46   max_position_embeddings: 1024
   47   micro_batch_size: 1
   48   global_batch_size: 32
   49
   50   # === 模型加载 ===
   51   load: /mnt/sfs_turbo/yzr/gyy_Asystem/Kimi-K2.5-mini
   52   tokenizer_model: /mnt/sfs_turbo/yzr/gyy_Asystem/Kimi-K2.5-mini
   53   tokenizer_type: HuggingFaceTokenizer
   54   trust_remote_code: true
   55   auto_detect_ckpt_format: true
   56   ckpt_format: torch_dist
   57
   58   # === 精度配置 (与 K2.5 1T 一致: BF16 + FP32 优化器) ===
   59   bf16: true
   60   attention_softmax_in_fp32: true
   61   use_precision_aware_optimizer: false
   62   main_grads_dtype: "fp32"
   63   main_params_dtype: "fp32"
   64   exp_avg_dtype: "fp32"
   65   exp_avg_sq_dtype: "fp32"
   66
   67   # === 优化器 ===
   68   optimizer: adam
   69   adam_beta1: 0.9
   70   adam_beta2: 0.999
   71   adam_eps: 1.0e-08
   72   lr: 2.0e-06
   73   lr_warmup_iters: 40
   74   clip_grad: 1.0
   75   weight_decay: 0.01
   76   use_distributed_optimizer: true
   77   data_parallel_sharding_strategy: no_shard   # 与 K2.5 1T 一致: DP=1 无分片
   78   train_iters: 10                             # 只跑几步, 采集显存即可
   79
   80   # === 激活重计算 (与 K2.5 1T 一致) ===
   81   recompute_granularity: full
   82   recompute_method: uniform
   83   recompute_num_layers: 1
   84
   85   # === 序列/通信并行 ===
   86   sequence_parallel: true
   87   overlap_grad_reduce: true
   88   overlap_p2p_comm: true
   89   deallocate_pipeline_outputs: false
   90   cp_comm_type: p2p
   91
   92   # === MoE 配置 ===
   93   moe_token_dispatcher_type: alltoall
   94   moe_grouped_gemm: true
   95   moe_permute_fusion: false
   96   moe_router_fusion: true
   97   moe_router_score_function: sigmoid
   98   moe_router_dtype: fp32
   99   moe_router_enable_expert_bias: true
  100   moe_router_bias_update_rate: 0.001
  101   moe_router_num_groups: 8
  102   moe_router_group_topk: 4
  103   moe_router_topk_scaling_factor: 2.5
  104   moe_per_layer_logging: true
  105   moe_shared_expert_overlap: false
  106
  107   # === Attention 配置 ===
  106
  107   # === Attention 配置 ===
  108   attention_backend: "flash"
  109   group_query_attention: true
  110   multi_latent_attention: true
  111   use_flash_attn: true
  112   apply_rope_fusion: true
  113   attention_dropout: 0.0
  114   hidden_dropout: 0
  115   normalization: RMSNorm
  116   norm_epsilon: 1e-05
  117   rotary_base: 10000000
  118   disable_bias_linear: true
  119
  120   # === 其他 ===
  121   gradient_accumulation_fusion: true
  122   use_mcore_models: true
  123   use_rotary_position_embeddings: true
  124   swiglu: true
  125   unidirectional: true
  126   distributed_backend: nccl
  127   distributed_timeout_minutes: 60
  128   enable_one_logger: false
  129   log_interval: 1
  130   log_throughput: true
  131   log_memory_to_tensorboard: true
  132   log_timers_to_tensorboard: true
  133   log_params_norm: true
  134   tensorboard_log_interval: 1
  135
  136   # === 日志级别 (详细显存日志) ===
  137   timing_log_level: 2
  138
  139   override_transformer_config:
  140     use_flash_attn: true
  141     recompute_method: uniform
  142     recompute_granularity: full
  143     recompute_num_layers: 1
  144     multi_latent_attention: true
  145     attention_mask_type: causal
  146     use_fused_rotary_pos_emb: true
  147     context_parallel_size: 2
  148
  149 loss_configs:
  150   adaptive_kl_horizon: 10000
  151   adaptive_kl_target: 6
  152   eps_clip: 0.2
  153   kl_ctl: 0.0
  154   temperature: 1
  155   token_normalize_scope: dp
  156
  157 asystem:
  158   enable: true
  159
  160 # === 显存快照采集 (核心功能) ===
  161 cuda_mem_track:
  162   enable: true
  163   steps: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 采集前 10 步, 观察碎片累积趋势
  164
  165 # torch_profile:
  166 #   steps: [1, 4]
  167 #   profile_memory: true
  168 #   use_gzip: true
  169
  170 upload_profile_to_ais_ckpt: false
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
```