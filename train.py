"""Baseline trainer. It WORKS and it is MEDIOCRE ON PURPOSE. Your hour goes
into changing what it does — schedule, init, optimizer, architecture,
tokenizer — inside the hard caps.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, total, peak_lr, warmup, floor_frac=0.1):
    """Linear warmup for `warmup` steps, then cosine decay to floor_frac*peak."""
    if step < warmup:
        return peak_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    coss = 0.5 * (1.0 + math.cos(math.pi * prog))
    return peak_lr * (floor_frac + (1.0 - floor_frac) * coss)


def make_optimizer(model, lr, weight_decay):
    # weight-decay the 2D matrices (Linear/embedding weights), not norms/biases
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--block", type=int, default=None, help="override cfg.block_size")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--qk_norm", action="store_true", help="RMS-norm q,k per head")
    ap.add_argument("--relu2", action="store_true", help="squared-ReLU MLP")
    ap.add_argument("--rmsnorm", action="store_true", help="RMSNorm instead of LayerNorm")
    ap.add_argument("--rope", action="store_true", help="rotary position embeddings")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    if args.block is not None:
        cfg.block_size = args.block
    cfg.qk_norm = args.qk_norm
    cfg.relu2 = args.relu2
    cfg.rmsnorm = args.rmsnorm
    cfg.rope = args.rope
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # Run 1: AdamW + warmup/cosine schedule + weight decay + gradient clipping
    opt = make_optimizer(model, args.lr, args.weight_decay)

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
