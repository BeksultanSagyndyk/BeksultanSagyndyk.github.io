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
<img width="627" alt="Screenshot 2024-02-06 at 14 43 59" src="https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/6ca75c46-0ab3-4910-b780-6bab0716a939">

FlashAttention - is not an approximate attention method ,its more about carefully accounting for reads and writes to different levels of fast and slow memory.
<img width="582" alt="Screenshot 2024-02-07 at 14 35 26" src="https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/f5a57483-e7a1-4ea2-97b2-ba3a02123382">

Main goal -  avoid reading and writing the attention matrix to and from HBM.
1) computing the softmax reduction without access to the whole input
2) not storing the large intermediate attention matrix for the backward pass.

How to get this?
1) restructure the attention computation to split the input into blocks and make several
passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling).
2) store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the
backward pass, which is faster than the standard approach of reading the intermediate attention matrix from
HBM.

# Background
GPU has Memory Hierarchy. 
GPUs have a massive number of threads to execute an operation (called a kernel).
Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

#### Depending on the balance of computation and memory accesses, operations can be classified as:

1) Compute-bound: the time taken by the operation is determined by how many arithmetic operations there
are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner
dimension, and convolution with large number of channels.
2) Memory-bound: the time taken by the operation is determined by the number of memory accesses, while
time spent in computation is much smaller. Examples include most other operations: elementwise (e.g.,
activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

#### Kernel fusion.
The most common approach to accelerate memory-bound operations is kernel fusion: if
there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of
multiple times for each operation.
