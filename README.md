# 📚 PyTorch-Tensor-100 挑战

<div align="center">
  <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="200px" alt="PyTorch Logo">
  <br>
  <h3>深入学习 PyTorch 张量操作的 100 个练习</h3>
</div>

> 本项目灵感来源于 [numpy-100](https://github.com/rougier/numpy-100) 项目。在当前开源社区中，我们发现缺少一个类似于 numpy-100 的专注于 PyTorch Tensor 操作的系统性练习题集，因此创建了这个项目，希望能填补这一空白。
> 这是一个为 PyTorch 初学者设计的循序渐进的训练题集。我们的目标是创建一个内容全面、知识点系统、并且**不仅仅聚焦于 Tensor 计算本身**的练习集。


## 🎯 总览

本项目包含 **100 道精心设计的核心练习题**，按照难度和主题分为六大模块：

| 模块 | 主题                      | 题目数量 | 难度   | 状态   |
| ---- | ------------------------- | -------- | ------ | ------ |
| 一   | Tensor 的创建与属性 ✨    | 9 题 | ⭐     | 完成 |
| 二   | 索引、切片与形状变换 🔪   | 26 题 | ⭐⭐   | 完成 |
| 三   | 数学与逻辑运算 🧮         | 15 题 | ⭐⭐   | 完成 |
| 四   | 广播机制 📡               | 15 题 | ⭐⭐⭐ | 未开始 |
| 五   | nn.Module 与基础层入门 🏗️ | 25 题 | ⭐⭐ | 未开始 |
| 六   | Autograd 自动求导引擎 🔥  | 20 题 | ⭐⭐⭐ | 未开始 |

---

## 📋 模块详解

### 模块一：Tensor 的创建与属性 ✨

**目标**: 掌握创建张量的各种方式，并了解其核心属性。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- 从 Python 列表 / NumPy 数组创建
- 创建常量张量：`zeros`, `ones`, `full`
- 创建随机张量：`rand`, `randn`, `randint`
- 创建序列张量：`arange`, `linspace`
- 获取核心属性：`shape`, `dtype`, `device`
- 仿照其他张量创建：`_like` 系列函数
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

- 通过简单示例（如矩阵加向量）直观感受广播
- 分析广播的规则和触发条件
- 一些需要利用广播机制的综合练习
</details>

### 模块五：nn.Module 与基础层入门 🏗️

**目标**: 将底层的 Tensor 操作与 `torch.nn` 中封装好的高级模块联系起来，为构建网络打下基础。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- **基础层**
  - 全连接层：`nn.Linear`
  - 激活函数层：`nn.ReLU`, `nn.Sigmoid` 等
  - 卷积层：`nn.Conv2d`
- **参数管理**
  - 查看模块参数：`.parameters()`
  - 参数初始化
  </details>

### 模块六：Autograd 自动求导引擎 🔥

**目标**: 理解 PyTorch 的核心——自动求导机制，这是训练神经网络的基石。

<details>
<summary><b>主要内容</b> (点击展开)</summary>

- 设置需要梯度：`requires_grad=True`
- 执行反向传播：`.backward()`
- 访问梯度：`.grad` 并理解梯度累加的特性
- 禁用梯度跟踪：`with torch.no_grad()` 或 `.detach()`
- 理解非标量输出的梯度计算
</details>

---

## 🏆 实战挑战（附加题）

**目标**: 在掌握所有基础技能后，通过解决源自经典论文的经典问题，综合运用并巩固所学知识。

<details>
<summary><b>示例项目</b> (点击展开)</summary>

- **Transformer**: 实现注意力掩码、位置编码
- **ViT**: 实现图像的patch提取部分
- **手动实现卷积**: 使用张量操作模拟 `nn.Conv2d`
</details>

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

推荐的Python版本: 3.9+
推荐的PyTorch版本: 1.10+

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
- [ ] 模块四：未开始
- [ ] 模块五：未开始
- [ ] 模块六：未开始
- [ ] 实战挑战：未开始

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
- **设计实战挑战题**: 我们特别欢迎基于经典论文设计的实战挑战题目:
  1. 从经典论文中提取核心算法或思想
  2. 将其转化为可实现的张量操作练习
  3. 提供清晰的问题描述和参考实现
  4. 在提交时注明论文出处和相关链接

### 贡献指南

- 确保新练习符合项目整体风格和结构
- 添加适当的注释，使代码易于理解
- 尽可能添加多样、全面的解答
- 实战挑战题目最好来源于经典论文的核心思想，帮助学习者理解深度学习领域的重要概念

如果您觉得这个项目对您有所帮助，请给它一个 ⭐️ 以示支持，这对维护者意义重大！
