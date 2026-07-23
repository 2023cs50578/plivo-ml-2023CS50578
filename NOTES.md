# NOTES — best configuration and why it works

1. **Best config:** byte-level **BPE (vocab 4096**, trained only on `train_corpus.txt`, with a lossless byte fallback) feeding a **1.91M-param tied-embedding GPT** (4 layers, n_embd 160, **block 256, RoPE, RMSNorm**, GELU MLP), trained **2000 steps** with **AdamW** (100-step warmup + cosine decay to 10%, weight decay 0.1, grad-clip 1.0), **batch 32**, **peak LR 2e-3**.
2. **The tokenizer is the dominant lever:** the corpus is ~1/3 Devanagari at 3 bytes/char, so a byte tokenizer burns the context window; BPE packs ~4 bytes/token, and because bits-per-byte divides summed per-token loss by a *fixed* byte count, quadrupling bytes/token slashes the score almost mechanically (2.24 → 2.03 from this change alone).
3. This is why the win is about **representation, not network size** — the same text carried by fewer, richer tokens is what lowers bpb.
4. Under the hard 2000-step cap the model is **token-starved, not capacity-starved**, so raising **batch 8 → 32** (4× more tokens/step and a cleaner gradient) was the single biggest *training* win (2.03 → 1.80).
5. A **warmup + cosine schedule with AdamW** beat the constant-LR baseline (2.37 → 2.24), and **peak LR 2e-3** was optimal (3e-3 overshot and got worse).
6. **Weight tying** became essential at vocab 4096 (it keeps the model under the 2M cap) even though it slightly *hurt* at vocab 256 — the parameter math flips with vocabulary size.
7. **RoPE** replaced the learned position table for a clean −2.6%, is **parameter-negative** (frees the 40,960-entry table), and lowered train loss too (learns faster); **RMSNorm** added a small further gain and also saved params.
8. **Longer context (block 128 → 256)** helped modestly (~0.5–1%) and was reserved for the final run because it triples CPU step time.
9. Several fashionable tricks **failed under the step cap and were deliberately excluded** — GPT-2 small-init + residual scaling, QK-Norm (tested twice, incl. as an LR-unlocker), and squared-ReLU (not robust across context lengths) — all tuned for long runs we don't have.
10. **Final dev bpb 1.6717**, a **29.5% reduction** from the 2.3718 baseline, at **1,891,360 params** and exactly 2000 steps.
