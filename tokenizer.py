"""Tokenizer: text <-> token ids.

Baseline was raw UTF-8 bytes (vocab 256), which spends 3 tokens on every
Devanagari character and burns the model's context window on the Hindi part
of the corpus. This replaces it with a byte-level BPE that LEARNS merges of
frequent adjacent pairs from train_corpus.txt only, compressing many bytes
into one token -> fewer, richer predictions over the same fixed byte count
-> lower bits-per-byte.

Interface kept identical (train.py / evaluate.py call load() with no args):
    load() -> obj with .encode(str)->list[int], .decode(list[int])->str, .vocab_size

Lossless guarantee: every id maps back to an exact byte string, so
decode(encode(text)) == text for arbitrary UTF-8. Anything the merges never
saw simply stays as raw-byte tokens (ids 0..255) -> byte fallback.

Train the merges (once), which writes bpe_merges.json next to this file:
    python tokenizer.py --train --data ../data/train_corpus.txt --vocab 4096
"""
import argparse
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
MERGES_PATH = os.path.join(HERE, "bpe_merges.json")

# GPT-2-style pre-tokenization: keep a leading space with a word, and treat
# runs of whitespace as their own chunk. Merges never cross these boundaries.
PAT = re.compile(r" ?[^\s]+|\s+")


class ByteTokenizer:
    """1 token == 1 byte. Guaranteed lossless. Used if no merges are trained."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="strict")


# ------------------------------ BPE training -------------------------------
def _apply_merge(w, a, b, new_id):
    out, i, n = [], 0, len(w)
    while i < n:
        if i < n - 1 and w[i] == a and w[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(w[i])
            i += 1
    return out


def train_bpe(text, vocab_size=4096):
    """Learn up to (vocab_size - 256) merges. ids 0..255 = raw bytes,
    256+ = learned merges (in order).

    Incremental: keeps a running pair->count and pair->{word indices}, and on
    each merge only re-counts the handful of words that actually contained the
    merged pair. A lazy max-heap picks the top pair (stale entries skipped).
    Trains the full 7 MB corpus in seconds instead of minutes.
    """
    assert vocab_size > 256
    chunk_freqs = Counter(m.group() for m in PAT.finditer(text))
    words = [list(c.encode("utf-8")) for c in chunk_freqs]
    freqs = list(chunk_freqs.values())

    counts = defaultdict(int)           # pair -> total weighted count
    where = defaultdict(set)            # pair -> set of word indices holding it
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p, k in Counter(zip(w, w[1:])).items():
            counts[p] += f * k
            where[p].add(wi)

    heap = [(-c, p) for p, c in counts.items()]
    heapq.heapify(heap)

    merges, next_id = [], 256
    for _ in range(vocab_size - 256):
        best = None
        while heap:
            negc, p = heapq.heappop(heap)
            if counts.get(p, 0) == -negc and counts[p] >= 2:
                best = p
                break
        if best is None:
            break
        a, b = best
        merges.append([a, b])
        dirty = set()
        for wi in list(where[best]):
            w = words[wi]
            f = freqs[wi]
            for p, k in Counter(zip(w, w[1:])).items():   # remove old contribution
                counts[p] -= f * k
                where[p].discard(wi)
                dirty.add(p)
            nw = _apply_merge(w, a, b, next_id)
            words[wi] = nw
            for p, k in Counter(zip(nw, nw[1:])).items():  # add new contribution
                counts[p] += f * k
                where[p].add(wi)
                dirty.add(p)
        for p in dirty:                                    # refresh heap lazily
            if counts.get(p, 0) > 0:
                heapq.heappush(heap, (-counts[p], p))
        next_id += 1
    return merges


# ------------------------------ BPE apply ----------------------------------
class BPETokenizer:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)
        self.id_to_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.id_to_bytes.append(self.id_to_bytes[a] + self.id_to_bytes[b])

    @lru_cache(maxsize=200_000)
    def _encode_chunk(self, chunk):
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            ids[best_i:best_i + 2] = [256 + best_rank]
        return ids

    def encode(self, text):
        out = []
        for m in PAT.finditer(text):
            out.extend(self._encode_chunk(m.group()))
        return out

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", "strict")


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py."""
    if os.path.exists(MERGES_PATH):
        return BPETokenizer(json.load(open(MERGES_PATH)))
    return ByteTokenizer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--data", default="../data/train_corpus.txt")
    ap.add_argument("--vocab", type=int, default=4096)
    args = ap.parse_args()
    if not args.train:
        raise SystemExit("pass --train")
    text = open(args.data, encoding="utf-8").read()
    merges = train_bpe(text, args.vocab)
    json.dump(merges, open(MERGES_PATH, "w"))
    tok = BPETokenizer(merges)
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round-trip FAILED (lossy!)"
    nb = len(text.encode("utf-8"))
    print(f"vocab_size={tok.vocab_size} (256 bytes + {len(merges)} merges)")
    print(f"bytes/token={nb/len(ids):.2f} (byte tokenizer=1.00)")
    print("round-trip OK (lossless)")


if __name__ == "__main__":
    main()
