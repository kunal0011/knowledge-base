# Domain IV: PyTorch — Deep Learning, Autograd & Tensor Compilation

[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.x%20Ready-red.svg)](https://pytorch.org/)
[![Foundational Book](https://img.shields.io/badge/Canonical%20Book-Deep%20Learning%20with%20PyTorch%20(Stevens)-darkred.svg)](https://pytorch.org/deep-learning-with-pytorch)
[![Architecture Paper](https://img.shields.io/badge/NeurIPS%20Paper-PyTorch%3A%20An%20Imperative%20Style-orange.svg)](https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727700-Paper.pdf)

> *"PyTorch was designed to bring the flexibility and pythonic intuition of dynamic computing to deep learning research and high-performance production systems. Through tape-based reverse-mode automatic differentiation, contiguous storage pointers, native CUDA stream integration, and modern graph compilation (`torch.compile`), PyTorch empowers engineers to train and serve models at unprecedented speed."*  
> — **Adam Paszke, Soumith Chintala et al.**, Creators of PyTorch

---

## 🏛️ PyTorch Architectural Philosophy & Memory Blueprint

A PyTorch `torch.Tensor` is fundamentally an object metadata wrapper around a 1D flat, contiguous memory chunk called a **`Storage`** (implemented in C++ as `THStorage` / `c10::StorageImpl`). Multiple tensors can reference the exact same underlying storage buffer by holding different offsets, shapes, and byte strides.

```
                         PyTorch Tensor Metadata & Storage Decoupling

  Tensor A (Shape: [2, 3], Strides: [3, 1], Offset: 0)
  +-------------------------------------------------------+
  | size: [2, 3] | stride: [3, 1] | offset: 0 | dtype     |
  +----+--------------------------------------------------+
       |
       | points to
       v
  +----+----+----+----+----+----+
  | 10 | 20 | 30 | 40 | 50 | 60 |  <-- 1D Flat Contiguous Storage Buffer (RAM or VRAM)
  +----+----+----+----+----+----+
       ^
       | points to (shared storage, different view)
  +----+--------------------------------------------------+
  | size: [3, 2] | stride: [1, 3] | offset: 0 | dtype     |
  +-------------------------------------------------------+
  Tensor B = Tensor A.t() (Transposed: Non-contiguous View!)
```

### The Dynamic Autograd Graph Engine
During the forward pass, PyTorch dynamically constructs a Directed Acyclic Graph (DAG) recording operations on tensors that have `requires_grad=True`. In the backward pass, the graph is traversed in reverse topological order:

```
    Tensor x (leaf, requires_grad=True)
       |
       v
  [ MulBackward0 ] <--- Tensor w (leaf, requires_grad=True)
       |
       v (h = x * w)
  [ AddBackward0 ] <--- Tensor b (leaf, requires_grad=True)
       |
       v (y = h + b)
  [ LossBackward ] <--- Ground Truth Target
       |
    loss.backward()  <--- Initiates reverse-mode AD graph traversal
```

---

## 📚 Master Chapter Index

| Chapter | Title | Core Canonical Concepts & Focus | Canonical Literature Focus |
| :---: | :--- | :--- | :--- |
| **01** | [Tensor Architecture, Storage, Strides & Memory Layout](01.%20Tensor%20Architecture%2C%20Storage%2C%20Strides%20%26%20Memory%20Layout.md) | `Storage` vs Tensor, strides, contiguity, views vs copies, pinned memory | *Deep Learning with PyTorch* Ch. 3 |
| **02** | [Autograd Mechanics, Computation Graph & Custom Functions](02.%20Autograd%20Mechanics%2C%20Computation%20Graph%20%26%20Custom%20Functions.md) | Tape autograd, `grad_fn`, dynamic DAG traversal, leaf tensors, custom ops | *Deep Learning with PyTorch* Ch. 5 |
| **03** | [Neural Network Modules, Parameters, Buffers & Hooks](03.%20Neural%20Network%20Modules%2C%20Parameters%2C%20Buffers%20%26%20Hooks.md) | `nn.Module` lifecycle, `Parameter` vs `Buffer`, forward/backward hooks | *Deep Learning with PyTorch* Ch. 6 |
| **04** | [Optimization Algorithms, Loss Functions & LR Schedulers](04.%20Optimization%20Algorithms%2C%20Loss%20Functions%20%26%20LR%20Schedulers.md) | SGD, Momentum, Adam, AdamW mechanics, schedulers, gradient clipping | *Deep Learning with PyTorch* Ch. 5 & Optim Lit |
| **05** | [Data Pipelines, Datasets, DataLoaders & Multiprocessing](05.%20Data%20Pipelines%2C%20Datasets%2C%20DataLoaders%20%26%20Multiprocessing.md) | `Dataset`, `IterableDataset`, `DataLoader` IPC, shared memory, custom collate | *Deep Learning with PyTorch* Ch. 7 |
| **06** | [Training Workflows, State Management & Checkpointing](06.%20Training%20Workflows%2C%20State%20Management%20%26%20Checkpointing.md) | Train vs eval mode, `state_dict`, atomic checkpointing, early stopping | *Programming PyTorch* Ch. 4 |
| **07** | [Mixed Precision Training (AMP) & Hardware Acceleration](07.%20Mixed%20Precision%20Training%20%28AMP%29%20%26%20Hardware%20Acceleration.md) | FP16/BF16, `torch.amp.autocast`, `GradScaler` underflow prevention, CUDA streams | *PyTorch Docs & NVIDIA Tensor Cores* |
| **08** | [Distributed Deep Learning (DDP, FSDP & Parallelism)](08.%20Distributed%20Deep%20Learning%20%28DDP%2C%20FSDP%20%26%20Parallelism%29.md) | `DistributedDataParallel`, NCCL Ring-AllReduce, FSDP, multi-node training | *PyTorch Distributed Architecture* |
| **09** | [PyTorch Compilation (TorchDynamo, AOTAutograd & Inductor)](09.%20PyTorch%20Compilation%20%28TorchDynamo%2C%20AOTAutograd%20%26%20Inductor%29.md) | `torch.compile`, bytecode capture, AOTAutograd, Triton code generation | *PyTorch 2.0 Paper & Core Docs* |
| **10** | [Model Deployment, Serialization, ONNX & Quantization](10.%20Model%20Deployment%2C%20Serialization%2C%20ONNX%20%26%20Quantization.md) | Serialization pitfalls, TorchScript JIT, ONNX export, dynamic/static/QAT | *Deep Learning with PyTorch* Ch. 15 |

---

## ⚡ Quick Reference: High-Performance PyTorch Idioms

### 1. Zero-Overhead Mixed Precision Training (`torch.amp`)
```python
import torch

scaler = torch.cuda.amp.GradScaler()

for data, target in dataloader:
    optimizer.zero_grad(set_to_none=True) # set_to_none saves memory bandwidth
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)
        
    # Scale loss to prevent FP16 gradient underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. PyTorch 2.0 Graph Compilation (`torch.compile`)
```python
import torch

# Compiles PyTorch graph down to fused Triton GPU kernels with zero Python overhead
model = MyNeuralNetwork()
compiled_model = torch.compile(model, mode="reduce-overhead")
```

### 3. Avoiding In-Place Modification on Autograd Leaves
Never perform in-place mutation (`+=`, `.add_()`) on variables required for backward graph derivative computations:
```python
# ANTI-PATTERN (Raises RuntimeError during backward: "variable modified in-place")
# x.add_(2.0)

# PRODUCTION IDIOM
x = x + 2.0
```
