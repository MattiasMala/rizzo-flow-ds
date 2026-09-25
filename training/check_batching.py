"""Check: right padding does not change the readout beyond BF16 shape noise.

A batch of eight copies of one row (no padding) already moves p by ~0.06 against the same row
alone (first run: 0.061, logits up to 0.375): matmul shapes change BF16 accumulation. So the
padded batches pass if they deviate no more than that reference, not if they are identical."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, Progress, load_model, log, read_jsonl
from train_lora import batches, readout_batch


def main():
    import torch

    rows = read_jsonl(DATA_DIR / "dev.jsonl")[:48]
    model = load_model("bf16").eval()
    worst, noise = 0.0, 0.0
    plan = batches(rows, 8, 8192)
    progress = Progress(len(plan), "check", every=1)
    with torch.inference_mode():
        for step, members in enumerate(plan, 1):
            chunk = [rows[i] for i in members]
            for row, batched in zip(chunk, readout_batch(model, chunk)):
                single = readout_batch(model, [row])[0]
                worst = max(worst, (batched.softmax(-1) - single.softmax(-1)).abs().max().item())
                copies = readout_batch(model, [row] * len(chunk))[0]
                noise = max(noise, (copies.softmax(-1) - single.softmax(-1)).abs().max().item())
            progress(step)
    ok = worst <= 1.5 * noise + 0.01
    log(
        f"max |dp| padded batch vs single {worst:.4f}, unpadded copies vs single {noise:.4f} -> {'OK' if ok else 'FAILED'}"
    )


if __name__ == "__main__":
    main()
