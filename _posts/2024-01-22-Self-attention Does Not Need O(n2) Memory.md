---
layout: post
title: "Paper Review 4: Self-attention Does Not Need O(n^2) Memory"
subtitle: "How can the memory efficiency of self-attention be increased?"
gh-repo: BeksultanSagyndyk/BeksultanSagyndyk.github.io
gh-badge: [star, follow]
tags: [LLM, efficiency, long input transformers]
comments: true
author: Beksultan Sagyndyk
---

![Screenshot 2024-01-26 at 17 56 17](https://github.com/BeksultanSagyndyk/BeksultanSagyndyk.github.io/assets/46630209/bac91768-4623-4516-bcdf-b381f8d20242)

In the paper "Self-attention Does Not Need O(n^2) Memory," the Google team introduces simple algorithms for attention and self-attention that require only constant memory and logarithmic memory, respectively. At a sequence length of 16384, the approach can reduce the self-attention memory overhead by 59x for inference and by 32x for differentiation.

## Intro

![Screenshot 2024-01-04 at 14 43 40](https://github.com/SanzharMrz/NLP-papers/assets/46630209/a4eacf1b-8f47-48db-b826-5df5b15587ae)

**Attention Algorithm:**
1. The attention operation on a single query produces a weighted sum of value vectors.
2. The weights are determined by the softmax of the dot products between the query and the keys.

In the implementation of this, we have to:
- First compute and remember S(i) for all i, -> an O(n) time and memory complexity for each query.
- Transformers use self-attention, meaning for each element of the sequence, we need our query -> time and space complexity is O(n^2).

## Main Algorithm

### Single Attention Case

**First Step:**

![Screenshot 2024-01-04 at 15 59 28](https://github.com/SanzharMrz/NLP-papers/assets/46630209/99f2b734-11c3-4660-9a1b-2c2427e851fd)

**Second Step:**
1. Initialize vectors v* and scalar s* with 0.
2. Loop over k_n = [k1, k2, ..kn] and v_n = [v1, v2, ...vn]. For each k_i and v_i, compute s_i.
3. For each s_i, update v* and s*.
4. After all, divide v*/s* -> final result.

![Screenshot 2024-01-04 at 15 57 59](https://github.com/SanzharMrz/NLP-papers/assets/46630209/fb5dc415-34fb-44f2-8473-a20b8d2e7c36)

### Self-Attention Case

To extend this algorithm to self-attention, just compute the results for all queries sequentially.

## Numerical Stability Problem

Default and new attention are not numerically stable when using floating-point arithmetic. For example, for scores ≥ 89, the exponentiation results in inf (for bfloat16 and float32).

To resolve this problem, they invented an additional scalar - m*:
1. Which keeps track of the maximum score that the incremental algorithm has seen so far.
2. Renormalize the sums of exponentiated values as needed.

![Screenshot 2024-01-04 at 16 21 35](https://github.com/SanzharMrz/NLP-papers/assets/46630209/28fdf34b-0223-480b-9930-39234cc79d8d)

## Results

![Screenshot 2024-01-04 at 16 19 11](https://github.com/SanzharMrz/NLP-papers/assets/46630209/af6bd29e-5851-44b5-b748-ed9634ee01a0)
