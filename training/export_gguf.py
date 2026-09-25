"""Merge a LoRA adapter into the BF16 checkpoint and export GGUF files llama.cpp can serve.

  .venv-train/Scripts/python training/export_gguf.py ADAPTER_DIR NAME [--quant q8_0]
  .venv-train/Scripts/python training/export_gguf.py --base NAME     # no adapter: pipeline check

Steps: merge (GPU, BF16) → save HF checkpoint → convert_hf_to_gguf.py of llama.cpp b11081 (the
commit of the pinned runtime) → llama-quantize. Outputs go to .research/merged/NAME/ and are
never overwritten. `--base` exports the untouched checkpoint, to compare with XHToken's GGUF.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DIR, ROOT, load_model, log

LLAMA_SRC = ROOT / ".research" / "llama.cpp-b11081"


def quantize_binary():
    """llama-quantize from any llama.cpp b11081 build installed by `rizzo download` (runtimes/)."""
    found = sorted((ROOT / "runtimes").glob("llama-b11081-*/llama-quantize*"))
    found = [p for p in found if p.suffix in ("", ".exe")]
    if not found:
        raise SystemExit(
            "llama-quantize not found in runtimes/: run `rizzo download --only runtime`"
        )
    return found[0]


COPY = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "configuration_spark.py",
    "modeling_spark.py",
    "LICENSE",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("adapter", nargs="?")
    parser.add_argument("name")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--quant", default="q8_0")
    args = parser.parse_args()
    if bool(args.adapter) == args.base:
        parser.error("give an adapter directory, or --base")
    out = ROOT / ".research" / "merged" / args.name
    out.mkdir(parents=True, exist_ok=False)
    hf = out / "hf"
    total = 4
    started = time.perf_counter()

    log(f"export step 1/{total}: merge {'nothing (base)' if args.base else args.adapter}")
    import torch

    model = load_model("bf16")
    if not args.base:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    log(
        f"export step 2/{total}: save HF checkpoint to {hf} (elapsed {time.perf_counter() - started:.0f}s)"
    )
    model.save_pretrained(hf, safe_serialization=True, max_shard_size="2GB")
    del model
    torch.cuda.empty_cache()
    for name in COPY:
        shutil.copy2(MODEL_DIR / name, hf / name)
    # tokenizer_config.json carries a minified copy of the chat template, which the converter
    # would prefer; without it the GGUF gets chat_template.jinja, like XHToken's own GGUF.
    # (The two render identical prompts: checked on all 2,000 typed-decisions questions.)
    config = json.loads((hf / "tokenizer_config.json").read_text(encoding="utf-8"))
    config.pop("chat_template", None)
    (hf / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bf16 = out / f"{args.name}-bf16.gguf"
    log(
        f"export step 3/{total}: convert to {bf16.name} (elapsed {time.perf_counter() - started:.0f}s)"
    )
    subprocess.run(
        [
            sys.executable,
            str(LLAMA_SRC / "convert_hf_to_gguf.py"),
            str(hf),
            "--outtype",
            "bf16",
            "--outfile",
            str(bf16),
        ],
        check=True,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(LLAMA_SRC / "gguf-py"),
            "PYTHONUTF8": "1",
        },
    )
    quantized = out / f"{args.name}-{args.quant}.gguf"
    log(
        f"export step 4/{total}: quantize to {quantized.name} (elapsed {time.perf_counter() - started:.0f}s)"
    )
    subprocess.run(
        [str(quantize_binary()), str(bf16), str(quantized), args.quant.upper()], check=True
    )
    log(f"done in {time.perf_counter() - started:.0f}s: {bf16}, {quantized}")


if __name__ == "__main__":
    main()
