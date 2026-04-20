# MindSpeed Patch 替换 Megatron-LM 机制详解

## 概述

MindSpeed 通过运行时动态替换机制，实现对 Megatron-LM 的无侵入式适配。用户只需一行代码即可启用所有补丁：

```python
import mindspeed.megatron_adaptor
```

---

## 1. 入口机制

### 1.1 主入口文件

**文件**: `mindspeed/megatron_adaptor.py`

```python
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

@AutoExecuteFunction  # 装饰器：import 时自动执行
def patch_features():
    """Patch all mindspeed related features."""
    global _IS_FEATURES_PATCHED
    if _IS_FEATURES_PATCHED:
        return
    _IS_FEATURES_PATCHED = True

    mindspeed_args = get_mindspeed_args()

    # 导入 Megatron 前的 patch
    MindSpeedFeaturesManager.apply_features_pre_patches(mindspeed_args)

    # 导入 Megatron 后的 patch
    MindSpeedFeaturesManager.apply_features_patches(mindspeed_args)


patch_features()  # 模块加载时自动执行
```

### 1.2 执行流程图

```
import mindspeed.megatron_adaptor
           │
           ▼
    patch_features()
           │
           ▼
    ┌─────────────────────────────────┐
    │ apply_features_pre_patches()    │  ← 导入 Megatron 前
    │ (如: 创建 dummy 模块)            │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │ apply_features_patches()        │  ← 导入 Megatron 后
    │ (如: 替换函数、类)               │
    └─────────────────────────────────┘
           │
           ▼
    遍历 FEATURES_LIST
           │
           ▼
    feature.register_patches()
           │
           ▼
    MindSpeedPatchesManager.apply_patches()
```

---

## 2. 核心 Patch 工具类

### 2.1 Patch 类

**文件**: `mindspeed/patch_utils.py`

单个 patch 的封装，负责：
- 解析目标函数路径
- 保存原始函数引用
- 应用/回滚 patch

```python
class Patch:
    def __init__(self, orig_func_name, new_func, create_dummy):
        # 解析路径: 'megatron.core.xxx.func' → module='megatron.core.xxx', func='func'
        split_name = orig_func_name.rsplit('.', 1)
        self.orig_module_name, self.orig_func_name = split_name
        self.orig_func = None          # 原始函数
        self.patch_func = new_func     # 替换函数
        self.wrappers = []             # 装饰器列表
        self.is_applied = False

    def apply_patch(self):
        """应用 patch: setattr(module, func_name, new_func)"""
        if self.is_applied:
            return

        # 获取目标模块和函数
        current_module, current_func = Patch.parse_path(
            self.orig_module_name, self.orig_func_name, self.create_dummy
        )

        if self.orig_module is None:
            self.orig_module, self.orig_func = current_module, current_func

        # 构建最终函数: patch_func + wrappers
        final_patch_func = self.orig_func
        if self.patch_func is not None:
            final_patch_func = self.patch_func

        for wrapper in self.wrappers:
            final_patch_func = wrapper(final_patch_func)

        # 核心替换: setattr
        if self.orig_func_name is not None:
            setattr(self.orig_module, self.orig_func_name, final_patch_func)

        # 同步更新已导入的模块引用
        for _, value in sys.modules.copy().items():
            if hasattr(value, self.orig_func_name) and \
               id(getattr(value, self.orig_func_name)) == id(current_func):
                setattr(value, self.orig_func_name, final_patch_func)

        self.is_applied = True
        self.final_patch_func = final_patch_func

    def remove_patch(self):
        """回滚 patch: 恢复原始函数"""
        for _, value in sys.modules.copy().items():
            if hasattr(value, self.orig_func_name) and \
               id(getattr(value, self.orig_func_name)) == id(self.final_patch_func):
                setattr(value, self.orig_func_name, self.orig_func)
        self.is_applied = False
```

### 2.2 MindSpeedPatchesManager 类

Patch 管理器，统一管理所有 patch 的注册和应用：

```python
class MindSpeedPatchesManager:
    patches_info: Dict[str, Patch] = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """
        注册 patch (不立即生效)

        参数:
            orig_func_name: 目标函数完整路径，如 'megatron.core.transformer.mlp.MLP.__init__'
            new_func: 替换函数
            force_patch: 是否强制覆盖已存在的 patch
            create_dummy: 目标不存在时是否创建 dummy 函数
        """
        if orig_func_name not in MindSpeedPatchesManager.patches_info:
            MindSpeedPatchesManager.patches_info[orig_func_name] = Patch(
                orig_func_name, new_func, create_dummy
            )
        else:
            MindSpeedPatchesManager.patches_info.get(orig_func_name).set_patch_func(
                new_func, force_patch
            )

    @staticmethod
    def apply_patches():
        """应用所有已注册的 patch"""
        for patch in MindSpeedPatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def remove_patches():
        """移除所有 patch，恢复原始函数"""
        for patch in MindSpeedPatchesManager.patches_info.values():
            patch.remove_patch()
            patch.remove_wrappers()
```

---

## 3. Patch 类型

### 3.1 类型对比

| 类型 | 函数名特征 | 行为 | 示例 |
|------|-----------|------|------|
| **直接替换** | 普通函数名 | 完全替换原函数 | `get_norm_tp_2d` |
| **装饰器包装** | 以 `wrapper`/`decorator` 结尾 | 包装原函数，可叠加 | `mindspeed_mlp_init_wrapper` |

### 3.2 直接替换

```python
from mindspeed.core.tensor_parallel.tp_2d.norm_factory_2d import get_norm_tp_2d

# 完全替换 megatron.legacy.model.utils.get_norm
patch_manager.register_patch(
    'megatron.legacy.model.utils.get_norm',
    get_norm_tp_2d
)
```

### 3.3 装饰器包装

```python
def mindspeed_mlp_init_wrapper(orig_init):
    """装饰器：包装原始 __init__ 方法"""
    def wrapper(self, *args, **kwargs):
        # 前置逻辑
        result = orig_init(self, *args, **kwargs)  # 调用原始方法
        # 后置逻辑
        self.some_new_attribute = ...
        return result
    return wrapper

# 包装原始方法 (函数名以 wrapper 结尾)
patch_manager.register_patch(
    'megatron.core.transformer.mlp.MLP.__init__',
    mindspeed_mlp_init_wrapper
)
```

### 3.4 装饰器叠加

```python
# 第一个装饰器
patch_manager.register_patch('megatron.xxx.func', wrapper1)
# 第二个装饰器 (叠加)
patch_manager.register_patch('megatron.xxx.func', wrapper2)

# 最终效果: func = wrapper2(wrapper1(orig_func))
```

---

## 4. Feature 插件架构

### 4.1 Feature 基类

**文件**: `mindspeed/features_manager/feature.py`

```python
class MindSpeedFeature:
    """MindSpeed 特性基类"""

    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.lower().strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0  # L0 特性默认启用

    def is_need_apply(self, args):
        """判断是否需要应用此特性"""
        return (self.optimization_level <= args.optimization_level
                and getattr(args, self.feature_name, None)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        """注册命令行参数"""
        pass

    def pre_validate_args(self, args: Namespace):
        """Megatron 参数验证前的预处理"""
        pass

    def post_validate_args(self, args: Namespace):
        """Megatron 参数验证后的恢复"""
        pass

    def validate_args(self, args: Namespace):
        """参数验证"""
        pass

    def pre_register_patches(self, patch_manager, args):
        """注册导入 Megatron 前的 patch"""
        pass

    def register_patches(self, patch_manager, args):
        """注册导入 Megatron 后的 patch"""
        pass

    def incompatible_check(self, global_args, check_args):
        """检查不兼容特性"""
        if getattr(global_args, self.feature_name, None) and \
           getattr(global_args, check_args, None):
            raise AssertionError(f'{self.feature_name} and {check_args} are incompatible.')

    def dependency_check(self, global_args, check_args):
        """检查依赖特性"""
        if getattr(global_args, self.feature_name, None) and \
           not getattr(global_args, check_args, None):
            raise AssertionError(f'{self.feature_name} requires {check_args}.')
```

### 4.2 Feature 实现示例

**文件**: `mindspeed/features_manager/tensor_parallel/tp_2d.py`

```python
class TP2dFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('tp-2d')  # 对应 --tp-2d 参数

    def register_args(self, parser: ArgumentParser):
        """注册命令行参数"""
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--tp-2d', action='store_true', default=False,
                           help='use 2d-tp to replace megatron-style tensor parallel')
        group.add_argument('--tp-x', type=int, default=1,
                           help='the first dim tensor parallel size')
        group.add_argument('--tp-y', type=int, default=1,
                           help='the second dim tensor parallel size')

    def validate_args(self, args):
        """参数验证"""
        self.incompatible_check(args, 'sequence_parallel')
        self.incompatible_check(args, 'use_fused_rmsnorm')

        if getattr(args, self.feature_name, None):
            if args.tensor_model_parallel_size // args.tp_x != args.tp_y:
                raise AssertionError('need satisfy tp = tp_x * tp_y')

    def register_patches(self, patch_manager, args):
        """注册所有 patch"""
        if not getattr(args, self.feature_name, None):
            return

        # 直接替换
        from mindspeed.core.tensor_parallel.tp_2d.norm_factory_2d import get_norm_tp_2d
        patch_manager.register_patch(
            'megatron.legacy.model.utils.get_norm',
            get_norm_tp_2d
        )

        # 装饰器包装
        from mindspeed.core.tensor_parallel.tp_2d.adaptor import mindspeed_mlp_init_wrapper
        patch_manager.register_patch(
            'megatron.core.transformer.mlp.MLP.__init__',
            mindspeed_mlp_init_wrapper
        )

        # 替换类
        from mindspeed.core.tensor_parallel.tp_2d.adaptor import MindSpeedRotaryEmbedding2D
        patch_manager.register_patch(
            'megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding',
            MindSpeedRotaryEmbedding2D
        )
```

### 4.3 特性分层

| 层级 | 名称 | 说明 | 默认启用 |
|------|------|------|---------|
| L0 | 基础兼容 | Megatron 基础适配 | 是 |
| L1 | 亲和增强 | NPU 算子优化 | 否 |
| L2 | 加速使能 | 高级并行策略 | 否 |

---

## 5. FeaturesManager 管理器

**文件**: `mindspeed/features_manager/features_manager.py`

```python
class MindSpeedFeaturesManager:
    FEATURES_LIST = []  # 所有注册的 Feature

    @classmethod
    def apply_features_pre_patches(cls, mindspeed_args):
        """应用所有 Feature 的 pre-patch"""
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mindspeed_args):
                feature.pre_register_patches(MindSpeedPatchesManager, mindspeed_args)
        MindSpeedPatchesManager.apply_patches()

    @classmethod
    def apply_features_patches(cls, mindspeed_args):
        """应用所有 Feature 的 patch"""
        for feature in cls.FEATURES_LIST:
            if feature.is_need_apply(mindspeed_args):
                feature.register_patches(MindSpeedPatchesManager, mindspeed_args)
        MindSpeedPatchesManager.apply_patches()

    @classmethod
    def remove_patches(cls):
        """移除所有 patch"""
        MindSpeedPatchesManager.remove_patches()
```

---

## 6. 使用示例

### 6.1 基本使用

```python
# 1. 导入 adaptor (自动应用所有 patch)
import mindspeed.megatron_adaptor

# 2. 正常使用 Megatron
from megatron.core.transformer import TransformerConfig
from megatron.training import pretrain

# 此时 Megatron 的函数已被 MindSpeed 替换
```

### 6.2 启用特定特性

```bash
python pretrain_gpt.py \
    --tp-2d \              # 启用 2D 张量并行
    --tp-x 4 \             # TP-X 维度
    --tp-y 2 \             # TP-Y 维度
    --tensor-model-parallel-size 8 \
    ...
```

### 6.3 动态重新 patch

```python
import mindspeed.megatron_adaptor

# 运行时动态修改参数并重新 patch
from mindspeed.megatron_adaptor import repatch
repatch({'tp_2d': True, 'tp_x': 4, 'tp_y': 2})
```

---

## 7. 设计优势

| 特点 | 说明 |
|------|------|
| **无侵入式** | 不修改 Megatron 源码，运行时动态替换 |
| **延迟生效** | `register_patch()` 只注册，`apply_patches()` 统一生效 |
| **可回滚** | `remove_patches()` 恢复所有原始函数 |
| **条件启用** | 通过命令行参数控制哪些 feature 生效 |
| **插件化** | Feature 独立封装，可组合使用 |
| **装饰器叠加** | 支持多层 wrapper，灵活扩展 |

---

## 8. 关键文件路径

```
MindSpeed/
├── mindspeed/
│   ├── megatron_adaptor.py          # 主入口
│   ├── patch_utils.py               # Patch 核心工具类
│   ├── features_manager/
│   │   ├── feature.py               # Feature 基类
│   │   ├── features_manager.py      # Feature 管理器
│   │   ├── tensor_parallel/
│   │   │   └── tp_2d.py             # TP-2D 特性实现
│   │   ├── transformer/
│   │   │   └── flash_attention/     # Flash Attention 特性
│   │   ├── pipeline_parallel/       # 流水线并行特性
│   │   └── moe/                     # MoE 特性
│   └── core/
│       ├── tensor_parallel/         # TP 核心实现
│       ├── transformer/             # Transformer 组件
│       └── ...
```

---

## 9. 常见 Patch 目标

| 目标路径 | 用途 |
|---------|------|
| `megatron.core.transformer.mlp.MLP.__init__` | MLP 层初始化 |
| `megatron.core.transformer.attention.SelfAttention.__init__` | Attention 层初始化 |
| `megatron.core.parallel_state.initialize_model_parallel` | 并行初始化 |
| `megatron.core.pipeline_parallel.schedules.*` | 流水线调度 |
| `megatron.training.arguments.validate_args` | 参数验证 |
| `megatron.core.models.common.embeddings.*` | 嵌入层 |