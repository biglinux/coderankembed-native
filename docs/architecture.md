# How coderankembed-native works

CodeRankEmbed is a 137-million-parameter BERT encoder (NomicBERT) trained on
the CoRNStack dataset: a text becomes the hidden state of its `[CLS]` token
after 12 layers. Each layer is one product for the queries, keys and values,
rotary position embeddings, attention, an output product, a SwiGLU
feed-forward of two more products, and two layer normalizations.

```text
text ─ tokenizer ─ ids ─ embeddings ─ 12 × [ qkv · rope · attention · out · norm · swiglu · norm ] ─ [CLS]
       src/tokenizer.rs   src/model.rs    src/model.rs (CPU) or src/gpu.rs + shaders/ (GPU)
```

The decisions that shape the code:

- **No padding.** A batch is its texts' tokens one after another. The
  products run over all of them at once and attention over each text alone,
  so a short text beside a long one costs nothing extra.
- **The last layer keeps one row a text.** Only `[CLS]` is pooled, so after
  the last attention the remaining products run over one row per text.
- **The CPU packs every weight once** into the 16-column panels its 6 × 16
  micro-kernel reads ([`src/gemm.rs`](../src/gemm.rs)), and attention packs
  each head's keys and values once, then runs both of its products through
  the same kernel. AVX2, AVX or SSE2 is chosen at run time, so one build
  runs on any x86-64.
- **The CPU threads are the crate's own pool**
  ([`src/pool.rs`](../src/pool.rs)): each thread keeps its scratch memory
  from one task to the next, and no task splits a sum, so results do not
  depend on the thread count. It takes about 300 lines, where `rayon` would
  be a dependency for the same work.
- **The GPU keeps the weights in half floats** and computes in floats: half
  the memory and bandwidth, with embeddings within 4e-4 of the reference's
  norm. On AMD and NVIDIA GPUs the large products also multiply in half
  floats, summing 8 products at a time before adding them in floats: 1.3 to
  1.6 times faster, and still within the parity tests (cosine above
  0.999996). Intel's driver runs that slower and less exactly, so it keeps
  floats. Attention never stores its score matrix: it runs the softmax
  online over tiles of 32 keys. A product of up to 32 rows, as a query's,
  runs `shaders/gemm_rows.comp`, which reads each weight once instead of
  filling the 128-row tiles of `shaders/gemm.comp` with rows that do not
  exist. A batch goes in one submission per layer at most, since a driver
  resets a GPU whose job runs past its timeout.
- **Vulkan is called directly** ([`src/vulkan/`](../src/vulkan)): the 49
  commands the engine uses are declared by hand and looked up in
  `libvulkan.so.1` at run time, so a machine without a Vulkan loader still
  runs the CPU path, and the crate needs no binding crate such as `ash`.
- **The tokenizer uses the Unicode tables Hugging Face's does**
  (`unicode-normalization` and `unicode_categories`), so accents, CJK and
  punctuation split the same way.
