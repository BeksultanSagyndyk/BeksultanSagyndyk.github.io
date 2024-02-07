---
layout: post
title: "Paper Review 7: FlashAttention - Fast and Memory-Efficient Exact Attention with IO-Awareness"
subtitle: "IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes"
gh-repo: BeksultanSagyndyk/BeksultanSagyndyk.github.io
gh-badge: [star, follow]
tags: [efficiency, long input transformers]
comments: true
author: Beksultan Sagyndyk
---

![Screenshot 2024-02-06](https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/6ca75c46-0ab3-4910-b780-6bab0716a939)

**FlashAttention** - is not an approximate attention method; it's more about carefully accounting for reads and writes to different levels of fast and slow memory.

![Screenshot 2024-02-07](https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/f5a57483-e7a1-4ea2-97b2-ba3a02123382)

**Main goal** - avoid reading and writing the attention matrix to and from HBM.
1. Computing the softmax reduction without access to the whole input.
2. Not storing the large intermediate attention matrix for the backward pass.

**How to achieve this?**
1. Restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling).
2. Store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM.

# Background
GPU has Memory Hierarchy. GPUs have a massive number of threads to execute an operation (called a kernel). Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

### Depending on the balance of computation and memory accesses, operations can be classified as:

1. Compute-bound: the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical examples are matrix multiply with a large inner dimension and convolution with a large number of channels.
2. Memory-bound: the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

### Kernel fusion
The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation.

### Standard Attention Implementation
![Screenshot 2024-02-07](https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/3195bcde-6e9c-478b-a70f-9349dc1ce4e7)

**Algorithm**
Require: Matrices Q, K, V ∈ R in HBM.
1. Load Q, K by blocks from HBM, compute S = QK>, write S to HBM.
2. Read S from HBM, compute P = softmax(S), write P to HBM.
3. Load P and V by blocks from HBM, compute O = PV, write O to HBM.
4. Return O.
