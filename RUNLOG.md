# RUNLOG — 2,000 Step LLM Speedrun

Scorer: `python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt`
Metric: dev bits-per-byte (bpb), lower is better. Cap: <=2000 steps, <=2,000,000 params, CPU.

| Run | Change (one thing) | dev bpb | params | note |
|----:|--------------------|--------:|-------:|------|
| 0 | Baseline, untouched starter | 2.3718 | 1,339,840 | constant-LR Adam, byte tokenizer, no tying |
| 1 | AdamW + warmup/cosine + weight decay + grad clip | 2.2447 | 1,339,840 | schedule only; model & tokenizer unchanged |
| 2 | init 0.02 + residual scaling + weight tying | 2.4664 | 1,298,880 | REVERTED — both hurt under step cap (isolated in 2a/2b) |
| 3 | BPE tokenizer vocab 4096 + weight tying (forced by cap) | 2.0296 | 1,913,280 | biggest single jump; 4.03 bytes/token |
| 4 | batch 8 -> 32 | 1.7956 | 1,913,280 | more tokens/step + less noise |
| 5 | + peak lr 1e-3 -> 2e-3 | 1.7266 | 1,913,280 | bigger batch tolerates higher LR |
| 6 | + block 128 -> 256 (at lr 1e-3) | 1.7653 | 1,933,760 | context helps but 3x slower; run at old LR |
| 7 | lr 2e-3 -> 3e-3 (block 128) | 1.7431 | 1,913,280 | overshoot; 2e-3 is the LR optimum |
| 8 | lr 2e-3 + block 256 (levers stacked) | 1.7155 | 1,933,760 | BEST; block256 only for the final run (3x slower) |
| 9 | + ReLU^2 MLP (block128 probe) | 1.7207 | 1,913,280 | small free win vs Run 5's 1.7266; keep |
| 10 | + QK-Norm (block128 probe) | 1.7600 | 1,913,600 | REJECTED — hurt at same LR |
| 11 | QK-Norm + lr 3e-3 (closure) | 1.7591 | 1,913,600 | REJECTED — QK-Norm does not unlock higher LR |
| 12 | block256 + lr2e-3 + ReLU^2 | 1.7386 | 1,933,760 | ReLU^2 helps at block128, hurts here -> not robust, dropped |
| 13 | **FINAL: block256 + lr2e-3 + GELU (no qk-norm, no relu2)** | **1.7155** | 1,933,760 | deliverable ckpt.pt (27.7% under baseline) |

---

## Run 0 — Baseline
**Hypothesis:** establish the number to beat.
**Changed:** nothing.
**Result:** dev bpb **2.3718**, 1,339,840 params, 2000 steps.
**Conclusion:** The starter is deliberately mediocre — constant LR with no warmup/schedule/weight-decay/clipping (train.py), no weight tying and a hot 0.05 init (model.py), and a byte-level tokenizer that spends 3 tokens per Devanagari character. Attack schedule first, then tokenizer.

## Run 1 — Optimizer + schedule
**Hypothesis:** under a 2000-step cap, a constant LR wastes both ends of training; a warmup+cosine schedule with AdamW should help without touching capacity.
**Changed (train.py only):** `Adam(constant 3e-4)` -> `AdamW(peak 1e-3, betas 0.9/0.95)`, linear warmup 100 steps, cosine decay to 10%, weight_decay 0.1 on 2D weights only, grad-clip 1.0.
**Result:** dev bpb 2.3718 -> **2.2447** (-5.4%). Train loss 1.73 -> 1.60.
**Conclusion:** Schedule matters a lot when steps are capped. Free win, no params added. Next: fix init + tie weights.

## Run 2 — Init + weight tying (FAILED, reverted)
**Hypothesis:** GPT-2-style init (0.02 + residual-branch scaling by 1/sqrt(2*n_layer)) and weight tying are standard best practice and should help.
**Changed (model.py):** init std 0.05 -> 0.02, added residual-proj scaling, tie_weights False -> True.
**Result:** dev bpb 2.2447 -> **2.4664 (worse)**. Because I bundled two changes, I isolated them from the Run 1 config:
  - Probe 2a (tying only, 0.05 init): 2.2731 — slightly worse.
  - Probe 2b (init only, no tying): 2.4087 — clearly worse.
**Conclusion:** Both standard tricks HURT under a 2000-step cap. The small-std init + residual scaling start activations too small to grow in only 2000 steps (train loss rose 1.60 -> 1.71), i.e. they trade early-speed for late-stability we can't use. Tying at vocab 256 saves only ~40k params but taxes accuracy. **Reverted to Run 1.** KEEP tying in reserve: at BPE vocab ~4096 it frees ~655k params worth reinvesting — re-test it there, not here.

## Run 3 — BPE tokenizer (the dominant lever)
**Hypothesis:** the score divides summed token-loss by a FIXED byte count. A byte tokenizer spends 3 tokens per Devanagari char; a BPE that packs ~4 bytes/token spreads the same information over 4x fewer, richer predictions and quadruples effective context -> large bpb drop.
**Changed (tokenizer.py + model.py):** byte tokenizer -> byte-level BPE, vocab 4096, trained on train_corpus.txt only, lossless byte fallback. Incremental trainer (lazy heap) does the full 7 MB in ~8 s. Turned on weight tying — now REQUIRED: untied at vocab 4096 the model is ~2.57M params (over cap); tied it is 1.913M.
**Result:** dev bpb 2.2447 -> **2.0296** (-9.6%; -14.4% vs baseline). Corpus 7.3M bytes -> 1.82M tokens (4.03 bytes/token). Params 1,913,280 < 2,000,000.
**Note:** per-token train loss (5.38) is NOT comparable to the byte-model loss — bigger vocab, harder per-token target. Only bpb is comparable. This is the whole reason the metric is per byte.
**Conclusion:** Tokenizer is the lever, exactly as the per-byte metric predicts. And Run 2's "failed" tying becomes correct here — the param math flipped. Next: feed more tokens/step (batch, context) since each step now sees only 1024 tokens.

## Run 4 — Batch size 8 -> 32
**Hypothesis:** at batch 8, each step sees only 1024 tokens (~1 epoch over 2000 steps) and a noisy gradient. A 4x larger batch gives a cleaner gradient and 4x more token exposure (~4.5 epochs).
**Changed (flag only):** --batch 8 -> 32.
**Result:** dev bpb 2.0296 -> **1.7956** (-11.5%). Train loss 5.38 -> 4.42. 99 ms/step (still ~3 min/run).
**Conclusion:** Big win — the model was token-starved, not capacity-starved. Next probe: higher peak LR (bigger batch tolerates it) and longer context.

## Run 5 — Peak LR 1e-3 -> 2e-3 (at batch 32)
**Hypothesis:** a larger batch has a lower-variance gradient, so it can safely take bigger steps; under a 2000-step cap a higher LR means more learning per step.
**Changed (flag only):** --lr 1e-3 -> 2e-3.
**Result:** dev bpb 1.7956 -> **1.7266** (-3.8%). CURRENT BEST.
**Conclusion:** Cheap, clean win. LR was the binding constraint after batch. Try pushing further (3e-3).

## Run 6 — Context 128 -> 256 (at lr 1e-3)
**Hypothesis:** BPE tokens are ~4 bytes, so block 128 only covers ~512 bytes of context; doubling it lets the model condition on more history.
**Changed (flag only):** --block 128 -> 256 (vs Run 4's lr 1e-3 baseline).
**Result:** dev bpb 1.7956 -> **1.7653** (-1.7%), but 288 ms/step (3x slower, ~10 min/run) and +20k params.
**Conclusion:** Context helps but modestly, and it's expensive. Worth combining with the good LR only if the final-run time budget allows. Next: test lr 3e-3 (cheap) and the block256+lr2e-3 combination.

## Run 7 — Peak LR 2e-3 -> 3e-3 (block 128)
**Hypothesis:** if 2e-3 helped, maybe 3e-3 helps more.
**Result:** 1.7266 -> **1.7431 (worse)**. LR optimum is ~2e-3; 3e-3 overshoots and destabilizes late training.
**Conclusion:** LR bracketed. 2e-3 is our value. (Architecture that stabilizes attention, e.g. QK-Norm, could raise this ceiling — worth a probe.)

## Run 8 — block 256 + lr 2e-3 (both good levers together)
**Hypothesis:** context (Run 6) and LR (Run 5) helped independently; combine them.
**Result:** **1.7155**, new best (vs Run 5 block128 1.7266). Params 1,933,760 < cap.
**Conclusion:** Levers stack. Keep block 256 for the FINAL run only (288 ms/step); iterate architecture probes at block 128 for speed, then apply winners to a block-256 final.

## Run 9 — Squared-ReLU MLP (block 128 probe)
**Hypothesis:** ReLU^2 is a zero-param swap that sometimes beats GELU in speedruns.
**Changed (model.py):** MLP activation GELU -> relu(x)^2.
**Result:** 1.7266 -> **1.7207** (-0.3%). Tiny but free (0 params).
**Conclusion:** Keep it. A free lottery ticket that paid a little.

## Run 10 — QK-Norm (block 128 probe, REJECTED)
**Hypothesis:** RMS-normalizing q,k per head stabilizes attention logits and should let us push LR past the 2e-3 ceiling that Run 7 hit.
**Changed (model.py):** added per-head RMSNorm on q and k before attention (+320 params).
**Result:** 1.7266 -> **1.7600 (worse)** at the same LR.
**Conclusion:** Wrong hypothesis. Forcing unit-RMS q/k discards magnitude information the small model actually uses, and 2000 steps is too few for the learnable gains to recover it. Same pattern as Run 2: a stability trick tuned for long runs is a net constraint under a hard step cap. Verified it also does not unlock a higher LR (Run 11). REJECTED.
