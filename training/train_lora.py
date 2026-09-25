"""LoRA (BF16 base) or QLoRA (NF4 base) on Spark-X2.5-4B, trained on the answer-letter readout.

The loss is the soft cross-entropy between the target distribution and the softmax of the
answer-letter logits at the last prompt position: the exact quantity the served model returns.
Nothing is generated and no other token is supervised. Batches are single sequences (the model
takes RoPE positions from `cache_position`, so right-padded batches would still be fine, but one
sequence per forward keeps the readout trivially correct) with gradient accumulation.

  .venv-train/Scripts/python training/train_lora.py --quant bf16 --r 16 --out .research/lora-runs/NAME
  ... --max-steps 20 --dev-limit 100      # smoke run
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, Progress, fmt, load_model, log, read_jsonl

TARGETS = ["q_k_v_proj", "g_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]


def readout(model, row):
    return readout_batch(model, [row])[0]


def readout_batch(model, rows):
    """Answer-letter logits of several right-padded prompts in one forward.

    Only the hidden state at each prompt's last real position is projected, and only onto the
    letter rows of the tied embedding: no full-vocabulary logits are materialized."""
    import torch

    width = max(len(r["tokens"]) for r in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device="cuda")
    mask = torch.zeros_like(ids)
    for i, r in enumerate(rows):
        ids[i, : len(r["tokens"])] = torch.tensor(r["tokens"], device="cuda")
        mask[i, : len(r["tokens"])] = 1
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    last = torch.tensor([len(r["tokens"]) - 1 for r in rows], device="cuda")
    h = hidden[torch.arange(len(rows), device="cuda"), last]
    # Tied checkpoint: the forward projects with the input embedding, so read the letters there.
    weight = (
        base.get_input_embeddings().weight
        if base.config.tie_word_embeddings
        else base.get_output_embeddings().weight
    )
    return [(h[i] @ weight[r["slots"]].T).float() for i, r in enumerate(rows)]


def soft_ce(logits, target):
    import torch

    t = torch.tensor(target, device=logits.device, dtype=torch.float32)
    return -(t * torch.log_softmax(logits, -1)).sum()


def batches(rows, batch, max_tokens, rng=None):
    """Length-bucketed batches: sort a shuffled pool by length, cut, shuffle the batches."""
    index = list(range(len(rows)))
    if rng:
        rng.shuffle(index)
    out = []
    pool = batch * 64
    for start in range(0, len(index), pool):
        chunk = sorted(index[start : start + pool], key=lambda i: len(rows[i]["tokens"]))
        current = []
        for i in chunk:
            width = max([len(rows[j]["tokens"]) for j in current] + [len(rows[i]["tokens"])])
            if current and (len(current) == batch or width * (len(current) + 1) > max_tokens):
                out.append(current)
                current = []
            current.append(i)
        if current:
            out.append(current)
    if rng:
        rng.shuffle(out)
    return out


def evaluate(model, rows, label, batch=8, max_tokens=8192):
    import torch

    model.eval()
    groups = defaultdict(lambda: [0.0, 0, 0])
    plan = batches(rows, batch, max_tokens)
    progress = Progress(len(plan), f"eval {label}", every=max(1, len(plan) // 4))
    with torch.inference_mode():
        for step, members in enumerate(plan, 1):
            chunk = [rows[i] for i in members]
            for row, logits in zip(chunk, readout_batch(model, chunk)):
                g = groups[row["group"].split("/")[0]]
                g[0] += soft_ce(logits, row["target"]).item()
                g[1] += int(
                    logits.argmax().item()
                    == max(range(len(row["target"])), key=row["target"].__getitem__)
                )
                g[2] += 1
            progress(step)
    model.train()
    total = [sum(g[i] for g in groups.values()) for i in range(3)]
    result = {"loss": total[0] / total[2], "acc": total[1] / total[2]}
    result |= {f"{k}_acc": g[1] / g[2] for k, g in sorted(groups.items())}
    log(f"eval {label}: " + " ".join(f"{k} {v:.4f}" for k, v in result.items()))
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, help="default 2*r")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=16, help="sequences per optimizer step")
    parser.add_argument("--batch", type=int, default=8, help="sequences per forward")
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=4096,
        help="padded tokens per forward (8192 ran out of memory)",
    )
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, help="optimizer steps; overrides --epochs")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int, default=600)
    parser.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    parser.add_argument("--log-every", type=int, default=5, help="optimizer steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="new run directory (never overwritten)")
    parser.add_argument("--resume", help="a checkpoint directory RUN/step-N written by this script")
    args = parser.parse_args()
    args.alpha = alpha = args.alpha or 2 * args.r

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.resume:
        resume = Path(args.resume)
        state = torch.load(resume / "trainer.pt", weights_only=False)
        saved = state["history"]["args"]
        keep = ("resume", "out", "eval_every", "log_every", "dev_limit")
        changed = {
            k: (saved.get(k), v)
            for k, v in vars(args).items()
            if k not in keep and saved.get(k) != v
        }
        if changed:
            raise SystemExit(f"--resume needs the original settings; differing: {changed}")
        out = resume.parent
    elif args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    else:
        parser.error("give --out for a new run or --resume for an interrupted one")
    train = read_jsonl(DATA_DIR / "train.jsonl")[: args.train_limit]
    dev = read_jsonl(DATA_DIR / "dev.jsonl")
    random.Random(1).shuffle(dev)
    dev = dev[: args.dev_limit]
    per_epoch = len(train) // args.grad_accum
    total_steps = args.max_steps or math.ceil(per_epoch * args.epochs)
    log(
        f"train {len(train)} rows, dev {len(dev)}, {total_steps} optimizer steps x {args.grad_accum} sequences"
    )

    mark = time.perf_counter()
    model = load_model(args.quant)
    if args.quant == "nf4":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    config = LoraConfig(
        r=args.r,
        lora_alpha=alpha,
        lora_dropout=args.dropout,
        target_modules=TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(
        f"{args.quant} base + LoRA r={args.r} alpha={alpha}: {trainable / 1e6:.1f}M trainable, loaded in {time.perf_counter() - mark:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(args.warmup * total_steps))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        return (
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        )

    history = {
        "args": vars(args) | {"alpha": alpha, "trainable": trainable, "rows": len(train)},
        "log": [],
        "eval": [],
    }
    # Preflight: forward + backward on the heaviest batches the run can draw, so an out-of-memory
    # shows up now and not hours in. Two worst cases: most padded tokens (activations) and most n*L^2 (eager attention scores).
    plan_all = batches(train, args.batch, args.max_batch_tokens)
    width = lambda b: max(len(train[i]["tokens"]) for i in b)
    candidates = {
        tuple(max(plan_all, key=lambda b: (len(b) * width(b), width(b)))),
        tuple(max(plan_all, key=lambda b: (len(b) * width(b) ** 2, len(b)))),
    }
    model.train()  # from_pretrained leaves eval mode, where gradient checkpointing is off
    for heavy in candidates:
        chunk = [train[i] for i in heavy]
        torch.cuda.reset_peak_memory_stats()
        sum(
            soft_ce(logits, row["target"])
            for row, logits in zip(chunk, readout_batch(model, chunk))
        ).backward()
        model.zero_grad(set_to_none=True)
        log(
            f"preflight: batch {len(chunk)} x {width(heavy)} tokens, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
        )
    rng = random.Random(args.seed)
    plan, cursor, first = [], 0, 1
    if args.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(resume / "adapter_model.safetensors"))
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        plan, cursor, first = state["plan"], state["cursor"], state["step"] + 1
        torch.set_rng_state(state["torch_rng"])  # LoRA dropout draws from these
        torch.cuda.set_rng_state(state["cuda_rng"])
        history = state["history"]
        log(f"resumed from {resume} at step {state['step']}/{total_steps}")
    else:
        history["eval"].append(
            {"step": 0} | evaluate(model, dev, "step 0", args.batch, args.max_batch_tokens)
        )
    model.train()
    progress = Progress(total_steps, "train", every=args.log_every, done=first - 1)
    window, seen_tokens = [], 0
    torch.cuda.reset_peak_memory_stats()
    for step in range(first, total_steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step - 1)
        loss_sum, taken = 0.0, 0
        while taken < args.grad_accum:
            if cursor >= len(plan):
                plan, cursor = batches(train, args.batch, args.max_batch_tokens, rng), 0
            chunk = [train[i] for i in plan[cursor]]
            cursor += 1
            losses = [
                soft_ce(logits, row["target"])
                for row, logits in zip(chunk, readout_batch(model, chunk))
            ]
            loss = sum(losses) / args.grad_accum
            loss.backward()
            loss_sum += sum(x.item() for x in losses)
            taken += len(chunk)
            seen_tokens += sum(len(r["tokens"]) for r in chunk)
        # Batches vary in size, so a step can take a few more than --grad-accum sequences:
        # rescale to the exact mean over the sequences actually used.
        for p in params:
            if p.grad is not None:
                p.grad.mul_(args.grad_accum / taken)
        loss_sum /= taken
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        window.append(loss_sum)
        if step % args.log_every == 0 or step == total_steps:
            mean = sum(window) / len(window)
            window = []
            elapsed = time.perf_counter() - progress.start
            history["log"].append(
                {"step": step, "loss": mean, "lr": lr_at(step - 1), "elapsed": elapsed}
            )
            progress(
                step,
                f"loss {mean:.4f} lr {lr_at(step - 1):.2e} {seen_tokens / elapsed:.0f} tok/s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
            )
        if step % args.eval_every == 0 and step != total_steps:
            history["eval"].append(
                {"step": step}
                | evaluate(model, dev, f"step {step}", args.batch, args.max_batch_tokens)
            )
            model.save_pretrained(out / f"step-{step}")
            # Everything --resume needs to continue exactly: optimizer moments, data order, RNG.
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "rng": rng.getstate(),
                    "plan": plan,
                    "cursor": cursor,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                out / f"step-{step}" / "trainer.pt",
            )
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    history["eval"].append(
        {"step": total_steps}
        | evaluate(model, dev, f"step {total_steps}", args.batch, args.max_batch_tokens)
    )
    model.save_pretrained(out / "final")
    history["seconds"] = history.get("seconds", 0) + time.perf_counter() - progress.start
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log(f"done in {fmt(history['seconds'])}, adapter in {out / 'final'}")


if __name__ == "__main__":
    main()
