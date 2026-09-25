"""Download everything the LoRA pipeline needs, at pinned revisions (run with .venv-train).

  python training/download_data.py              # all: model, training data, benchmark, sources
  python training/download_data.py --only data   # model | data | bench | sources

Existing files with the expected size are kept; interrupted downloads resume (HTTP Range).
Weight files are checked against the sha256 Hugging Face publishes for them (LFS oid).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, MODEL_DIR, ROOT, fmt, log

HF = "https://huggingface.co"
MODEL = ("XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4")
TYPED = ("LocalLLaMA/typed-decisions", "c76749ec58bd8c3d2ea706b31c333a9059c38f90")
TASKSOURCE = ("tasksource/procedural-typed-decisions", "b4f55bea135283127b2945c8e6fc7e68e9c56a4a")
OPENJEV = ("ZefanCai/Open-Jev", "c67699e13d0ae25e35b77165a4b6b079bedc8aba")
JEVBENCH = ("Praveenrajus/jev-bench", "b41e6b2f68a429608e1c0d354324d0bebaec7971")
SEMIF = ("https://github.com/TheoLeeCJ/SemIf.git", "ca3ba65f142967030ecb453346e94d6f476a69df")
LLAMA = (
    "https://github.com/ggml-org/llama.cpp.git",
    "b11081",
    "161755f29e415e2c33efe906e91843c068efd664",
)

TASKSOURCE_CONFIGS = [
    "arithmetic",
    "entity_belief_tracking",
    "event_state_reconstruction",
    "evidence_sufficiency",
    "multi_view_adjudication",
    "needle_retrieval",
    "partial_observation_calibration",
    "policy_applicability",
    "record_aggregation",
    "state_perturbation",
    "table_lookup",
]
JEVBENCH_CONFIGS = [
    "civil_comments",
    "strategyqa_closed",
    "strategyqa_grounded",
    "paws",
    "helpsteer2_helpfulness",
    "helpsteer2_verbosity",
    "measuring_hate_speech",
    "mmlu",
    "arc_challenge",
    "boolq",
    "fever_evidence",
    "stsb",
]


def fetch(url, dest, size=None, sha256=None, label=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is not None and have == size:
        log(f"{label} {dest.name}: present")
    else:
        if size is None and have:
            have = 0  # unknown size: download again rather than trust a partial file
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(request) as response:
            total = have + int(response.headers.get("Content-Length", 0))
            mode = "ab" if have and response.status == 206 else "wb"
            done = have if mode == "ab" else 0
            start, last = time.perf_counter(), 0.0
            with open(dest, mode) as stream:
                while chunk := response.read(1 << 20):
                    stream.write(chunk)
                    done += len(chunk)
                    now = time.perf_counter()
                    if now - last > 10 and total:
                        rate = (done - have) / (now - start)
                        log(
                            f"{label} {dest.name}: {done / 2**20:.0f}/{total / 2**20:.0f} MiB ETA {fmt((total - done) / rate) if rate else '?'}"
                        )
                        last = now
        log(f"{label} {dest.name}: downloaded {dest.stat().st_size / 2**20:.1f} MiB")
    if sha256:
        digest = hashlib.sha256()
        with open(dest, "rb") as stream:
            while chunk := stream.read(1 << 24):
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise SystemExit(f"sha256 mismatch for {dest}: delete it and run again")
        log(f"{label} {dest.name}: sha256 ok")


def model():
    repo, rev = MODEL
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/{rev}") as response:
        files = [f for f in json.load(response) if f["type"] == "file"]
    for step, f in enumerate(files, 1):
        lfs = f.get("lfs") or {}
        fetch(
            f"{HF}/{repo}/resolve/{rev}/{f['path']}",
            MODEL_DIR / f["path"],
            f["size"],
            lfs.get("oid"),
            f"model {step}/{len(files)}",
        )


def data():
    raw = DATA_DIR / "raw"
    jobs = []
    for c in TASKSOURCE_CONFIGS:
        for s in ("train", "validation"):
            jobs.append(
                (
                    f"{HF}/datasets/{TASKSOURCE[0]}/resolve/{TASKSOURCE[1]}/{c}/{s}-00000-of-00001.parquet",
                    raw / "tasksource" / f"{c}__{s}.parquet",
                )
            )
    for s in ("train", "validation"):
        jobs.append(
            (
                f"{HF}/datasets/{OPENJEV[0]}/resolve/{OPENJEV[1]}/data/release-v2-redistributable/{s}-00000-of-00001.parquet",
                raw / "openjev" / f"{s}.parquet",
            )
        )
    for c in JEVBENCH_CONFIGS:
        for s in ("train", "validation"):
            jobs.append(
                (
                    f"{HF}/datasets/{JEVBENCH[0]}/resolve/{JEVBENCH[1]}/data/{c}/{s}.jsonl",
                    raw / "jevbench" / f"{c}__{s}.jsonl",
                )
            )
    for step, (url, dest) in enumerate(jobs, 1):
        if dest.exists() and dest.stat().st_size:
            continue
        fetch(url, dest, label=f"data {step}/{len(jobs)}")
    log(f"data: {len(jobs)} files in {raw}")


def bench():
    import pyarrow.parquet as pq

    out = ROOT / ".research" / "typed-decisions" / "all"
    for step, split in enumerate(("test", "train"), 1):
        parquet = out / f"{split}.parquet"
        fetch(
            f"{HF}/datasets/{TYPED[0]}/resolve/{TYPED[1]}/all/{split}-00000-of-00001.parquet",
            parquet,
            label=f"bench {step}/2",
        )
        rows = pq.read_table(parquet).to_pylist()
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        log(f"bench {split}: {len(rows)} rows -> {split}.jsonl")


def sources():
    semif = ROOT / ".research" / "SemIf"
    if not semif.exists():
        subprocess.run(["git", "clone", "-q", SEMIF[0], str(semif)], check=True)
    subprocess.run(["git", "-C", str(semif), "checkout", "-q", SEMIF[1]], check=True)
    log(f"SemIf at {SEMIF[1][:7]} (its fixtures are excluded from training)")
    llama = ROOT / ".research" / f"llama.cpp-{LLAMA[1]}"
    if not llama.exists():
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "--branch", LLAMA[1], LLAMA[0], str(llama)],
            check=True,
        )
    head = subprocess.run(
        ["git", "-C", str(llama), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != LLAMA[2]:
        raise SystemExit(f"{llama} is at {head}, expected {LLAMA[2]}")
    log(f"llama.cpp source at {LLAMA[1]} ({head[:7]}), used only by export_gguf.py")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", choices=("model", "data", "bench", "sources"))
    args = parser.parse_args()
    steps = [("model", model), ("data", data), ("bench", bench), ("sources", sources)]
    steps = [s for s in steps if not args.only or s[0] == args.only]
    for number, (name, run) in enumerate(steps, 1):
        log(f"== step {number}/{len(steps)}: {name}")
        run()


if __name__ == "__main__":
    main()
