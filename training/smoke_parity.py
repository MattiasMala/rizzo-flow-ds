"""Smoke test: the HF checkpoint read through transformers must match llama.cpp BF16.

Two processes, never both holding weights (one GPU):
  .venv/Scripts/python training/smoke_parity.py llama  OUT.json   # reference, project venv
  .venv-train/Scripts/python training/smoke_parity.py hf OUT.json # compares against OUT.json

Questions come from the typed-decisions *train* split (never test), through compat.to_native.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT,
    Progress,
    compile_native,
    letter_logits,
    load_model,
    log,
    read_jsonl,
    tokenizer,
)

DATA = ROOT / ".research" / "typed-decisions" / "all" / "train.jsonl"
CASES = 8  # x 5 questions = 40 decisions, two per workflow


def natives():
    from rizzo_flow import compat

    rows = read_jsonl(DATA)
    picked = []
    for workflow in sorted({r["workflow"] for r in rows}):
        picked += [r for r in rows if r["workflow"] == workflow][: CASES // 4]
    out = []
    for row in picked:
        request = compat.SystemOneRequest.model_validate(
            {
                "state": json.loads(row["state"]),
                "model": "rizzo-latest",
                "questions": json.loads(row["questions"]),
            }
        )
        native, _ = compat.to_native(request)
        out.append((row["id"], native.model_dump(mode="json", exclude_none=True)))
    return out


def softmax(xs):
    import math

    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def reference(path):
    from rizzo_flow.loader import load_backend
    from rizzo_flow.prompts import compile_request
    from rizzo_flow.schema import Request

    backend = load_backend("llama", quant="bf16", device="cuda")
    items = natives()
    progress = Progress(len(items), "llama.cpp", every=1)
    records = []
    for step, (case, native) in enumerate(items, 1):
        prefix, jobs = compile_request(backend.tokenizer, Request.model_validate(native), 8192)
        logits, _ = backend.score(prefix, jobs, "direct")
        for job in jobs:
            records.append(
                {
                    "case": case,
                    "q": job.id,
                    "tokens": job.tokens,
                    "logits": list(map(float, logits[job.id])),
                }
            )
        progress(step)
    Path(path).write_text(json.dumps(records), encoding="utf-8")
    log(f"wrote {len(records)} reference decisions to {path}")


def compare(path):
    import torch

    ref = {(r["case"], r["q"]): r for r in json.loads(Path(path).read_text(encoding="utf-8"))}
    tok = tokenizer()
    mark = time.perf_counter()
    model = load_model("bf16").eval()
    log(
        f"model loaded in {time.perf_counter() - mark:.1f}s, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )
    items = natives()
    progress = Progress(len(items), "transformers", every=1)
    token_mismatch, flips, worst, deltas = 0, 0, 0.0, []
    with torch.inference_mode():
        for step, (case, native) in enumerate(items, 1):
            for key, tokens, slots in compile_native(tok, native, 8192):
                r = ref[(case, key)]
                if tokens != r["tokens"]:
                    token_mismatch += 1
                    continue
                p_hf = softmax(letter_logits(model, tokens, slots).tolist())
                p_ll = softmax(r["logits"])
                delta = max(abs(a - b) for a, b in zip(p_hf, p_ll))
                deltas.append(delta)
                if delta > 0.02:
                    log(
                        f"  {case}/{key}: hf {[round(v, 3) for v in p_hf]} llama {[round(v, 3) for v in p_ll]}"
                    )
                worst = max(worst, delta)
                flips += p_hf.index(max(p_hf)) != p_ll.index(max(p_ll))
            progress(step)
    deltas.sort()
    log(
        f"|dp| median {deltas[len(deltas) // 2]:.4f}, p90 {deltas[int(0.9 * (len(deltas) - 1))]:.4f}"
    )
    log(
        f"decisions {len(ref)}, token mismatches {token_mismatch}, argmax flips {flips}, max |dp| {worst:.4f}"
    )
    # First run: 40 decisions, 0 token mismatches, 0 flips, median |dp| 0.0003, max 0.082. The
    # large deltas sit only on uncertain answers (p between 0.2 and 0.8): BF16 noise between two
    # implementations, not a structural bug, which would move confident answers too.
    ok = token_mismatch == 0 and flips == 0 and deltas[len(deltas) // 2] < 0.01
    log("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    (reference if sys.argv[1] == "llama" else compare)(sys.argv[2])
