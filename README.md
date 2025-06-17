# 📚 PyTorch-Tensor-100 挑战

<div align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="200px" alt="PyTorch Logo">
  <br>
  <h3>从张量到模型的 100 个 PyTorch 练习</h3>
</div>

> 本项目灵感来源于 [numpy-100](https://github.com/rougier/numpy-100) 项目。在当前开源社区中，我们发现缺少一个类似于 numpy-100 的系统性 PyTorch 学习路径，因此创建了这个项目，希望能填补这一空白。
> 这是一个为 PyTorch 初学者设计的循序渐进的训练题集。我们的目标是创建一个内容全面、知识点系统的练习集，**从基础张量操作到实际应用预训练模型**，提供完整的 PyTorch 入门学习体验。

## 🎯 总览

本项目包含 **100 道精心设计的核心练习题**，按照难度和主题分为六大模块，形成从张量基础知识到实际模型应用的完整学习路径：

| 模块 | 主题                            | 题目数量 | 难度   | 状态   |
| ---- | ------------------------------- | -------- | ------ | ------ |
| 一   | Tensor 的创建与属性 ✨          | 9 题     | ⭐     | 完成   |
| 二   | 索引、切片与形状变换 🔪         | 26 题    | ⭐⭐   | 完成   |
| 三   | 数学与逻辑运算 🧮               | 15 题    | ⭐⭐   | 完成   |
| 四   | 广播机制 📡                     | 15 题    | ⭐⭐⭐ | 完成   |
| 五   | nn.Module 与基础层入门 🏗️       | 20 题    | ⭐⭐   | 未开始 |
| 六   | torchvision 与预训练模型应用 🖼️ | 15 题    | ⭐⭐⭐ | 未开始 |

---

## 📋 模块详解

### 模块一：Tensor 的创建与属性 ✨

**目标**: 掌握创建张量的各种方式，并了解其核心属性。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **创建张量**
  - 从 Python 列表 / NumPy 数组创建：`tensor()`
  - 创建常量张量：`zeros`, `ones`, `full`, `eye`
  - 创建随机张量：`rand`, `randn`, `randint`, `normal`
  - 创建序列张量：`arange`, `linspace`, `logspace`
- **张量属性**
  - 获取核心属性：`shape`, `dtype`, `device`, `requires_grad`
  - 类型转换：`to()`, `type()`
  - 仿照其他张量创建：`zeros_like`, `ones_like`, `full_like` 等
- **设备管理**
  - CPU/GPU 切换：`to('cuda')`, `to('cpu')`
  - 检查可用设备：`torch.cuda.is_available()`
  </details>

### 模块二：索引、切片与形状变换 🔪

**目标**: 学习如何灵活地访问、重塑、组合和拆分张量，这是数据处理的核心。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **索引与切片**
  - 基础索引与切片：`[]`
  - 布尔索引：`[True or False]`
  - 高级索引：`index_select`, `gather`
- **形状操作**
  - 改变形状：`view`, `reshape`
  - 交换维度：`permute`, `transpose`
  - 维度操作：`unsqueeze`, `squeeze`
  - 扩展与复制：`expand`, `repeat`
- **组合与拆分**
  - 拼接：`cat`, `stack`
  - 分割：`split`, `chunk`
  - 条件选择与填充：`where`, `scatter_`, `index_put_`, `masked_fill_`
- **内存布局**
  - 连续性：`contiguous`
- **高级应用**
  - 对角线提取：`diagonal`
  - 序列翻转：`flip`
  - 网格生成：`meshgrid`
  </details>

### 模块三：数学与逻辑运算 🧮

**目标**: 掌握张量的数值计算，这是构建神经网络前向传播的基础。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **基础运算**
  - 逐元素运算：加、减、乘、除、幂等
  - 矩阵运算：矩阵乘法 (`matmul` 或 `@`)
- **聚合操作**
  - 求和：`sum`
  - 均值：`mean`
  - 最大/最小值：`max`, `min`
  - 标准差：`std`
  - 结合 `dim` 和 `keepdim` 参数
- **逻辑与比较**
  - 比较运算符：`>`, `<`, `==`
  - 逻辑函数：`all`, `any`
  </details>

### 模块四：广播机制 📡

**目标**: 深入理解当两个形状不同的张量进行运算时，PyTorch 如何自动扩展维度来匹配形状。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **广播基础**
  - 广播规则与原理解析
  - 矩阵与向量、矩阵与标量的运算
  - 不同维度张量间的自动扩展
- **广播应用**
  - 批量数据处理中的广播技巧
  - 使用广播加速计算并节省内存
  - 常见广播陷阱与解决方案
- **高级应用**
  - 在神经网络中的实际应用（如注意力机制）
  - 结合其他操作（如 `unsqueeze`）实现复杂广播
  - 广播与并行计算的关系
  </details>

### 模块五：nn.Module 与基础层入门 🏗️

**目标**: 将底层的 Tensor 操作与 `torch.nn` 中封装好的高级模块联系起来，为构建网络打下基础。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **基础层**
  - 全连接层：`nn.Linear` - 实现线性变换
  - 激活函数层：`nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.LeakyReLU`, `nn.GELU` 等
  - 卷积层：`nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d` - 不同维度的卷积操作
  - 池化层：`nn.MaxPool2d`, `nn.AvgPool2d` - 降采样操作
  - 归一化层：`nn.BatchNorm2d`, `nn.LayerNorm` - 稳定训练
  - Dropout 层：`nn.Dropout` - 防止过拟合
- **模块构建**
  - 创建自定义模块：继承 `nn.Module`
  - 实现 `__init__` 和 `forward` 方法
  - 模块嵌套与复用
  - 使用 `nn.Sequential` 构建简单网络
- **参数管理**
  - 查看模块参数：`.parameters()`, `.named_parameters()`
  - 参数初始化：`nn.init` 系列函数
  - 参数共享与冻结
  - 保存与加载模型参数：`torch.save`, `torch.load`
- **函数式 API**
  - 使用 `F` 模块中的函数版本
  - 模块与函数的对应关系
- **实用技巧**
  - 模型结构可视化
  - 参数量计算
  - 模型调试与检查
  - 处理不同设备上的模型
  </details>

### 模块六：Autograd 自动求导引擎 🔥

**目标**: 理解 PyTorch 的核心——自动求导机制，这是训练神经网络的基石。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **自动求导基础**
  - 设置需要梯度：`requires_grad=True`
  - 执行反向传播：`.backward()`
  - 访问梯度：`.grad` 并理解梯度累加的特性
  - 梯度清零：`.grad = None` 或 `.zero_grad()`
  - 计算图与动态图机制
- **梯度控制**
  - 禁用梯度跟踪：`with torch.no_grad()` 或 `.detach()`
  - 保留计算图：`retain_graph=True`
  - 设置梯度不可变：`.requires_grad_(False)`
  - 使用 `torch.enable_grad()` 和 `torch.set_grad_enabled()`
- **高级应用**
  - 理解非标量输出的梯度计算与 Jacobian 矩阵
  - 自定义自动求导函数：使用 `torch.autograd.Function`
  - 二阶导数与 Hessian 矩阵计算
  - 检测梯度异常：处理梯度爆炸和梯度消失
- **性能优化**
  - 使用 `torch.inference_mode()` 提升推理性能
  - 减少内存使用：梯度检查点 (gradient checkpointing)
  - 自动混合精度训练中的梯度缩放
  - 分析与调试自动求导性能瓶颈
  </details>

### 模块六：torchvision 与预训练模型应用 🖼️

**目标**: 学习如何利用 torchvision 库加载预训练模型，处理图像数据，并进行高效推理，应用前面所学的张量知识到实际场景。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **预训练模型使用**
  - 加载经典模型：ResNet, VGG, EfficientNet, YOLO
  - 模型权重管理：`torch.hub`和`torchvision.models`
  - 模型结构探索与修改
  - 高效推理：使用`torch.inference_mode()`和`torch.no_grad()`
- **图像处理与变换**
  - 图像加载与预处理：`torchvision.io`和`transforms`
  - 数据增强技术：裁剪、翻转、颜色变换
  - 批量处理与归一化
  - 自定义转换流程
- **视觉任务实践**
  - 图像分类：使用 ImageNet 预训练模型
  - 目标检测：使用 COCO 预训练模型
  - 图像分割：使用预训练分割模型
  - 特征提取与迁移学习
- **结果评估与可视化**
  - 推理结果处理与后处理
  - 性能评估指标：准确率、IoU 等
  </details>

---

## 🏆 更多深度学习实践

想要更多实战挑战？请访问我的另一个项目 [PyTorch 模型手写实现集合](https://github.com/liuxiang09/Pytorch)，其中包含了多个经典深度学习模型的完整实现，包括但不限于：

- ResNet
- VGG
- Transformer
- CLIP
- 等更多高级模型

这些实现将帮助你将基础知识应用到实际模型构建中，进一步提升你的 PyTorch 技能。

## 🛠️ 环境配置

要运行本项目中的练习，您需要配置以下环境：

```bash
# 使用pip安装依赖
pip install torch torchvision torchaudio matplotlib jupyter

# 或使用conda
conda create -n pytorch-tensor-100 python=3.10
conda activate pytorch-tensor-100
conda install pytorch torchvision torchaudio matplotlib jupyter -c pytorch
```

推荐的 Python 版本: 3.10+

您可以通过运行以下代码验证环境是否正确配置：

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
```

## 📊 项目进度追踪

- [x] 模块一：完成 9/9 题
- [x] 模块二：完成 26/26 题
- [x] 模块三：完成 15/15 题
- [x] 模块四：完成 15/15 题
- [ ] 模块五：未开始
- [ ] 模块六：未开始

## 📝 学习建议

- 每个练习都尝试自己实现，然后对比参考答案
- 理解每个操作背后的原理，而不只是记住 API
- 尝试将多个操作组合起来解决更复杂的问题
- 使用 PyTorch 官方文档作为补充学习资源

## 🤝 参与贡献

我们非常欢迎并感谢您对本项目做出贡献，无论贡献大小！

### 如何贡献

- **提交 Issue**: 发现错误、有改进建议或新的练习想法? [提交 Issue](../../issues/new) 告诉我们!
- **提交 Pull Request**: 直接修复错误或添加新功能:
  1. Fork 本仓库
  2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
  3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
  4. 推送到分支 (`git push origin feature/AmazingFeature`)
  5. 打开 Pull Request

### 贡献指南

- 确保新练习符合项目整体风格和结构
- 添加适当的注释，使代码易于理解
- 尽可能添加多样、全面的解答

如果您觉得这个项目对您有所帮助，请给它一个 ⭐️ 以示支持，这对维护者意义重大！
