"""Build the LoRA training mix: one row per (state, question), pre-tokenized with the served prompt.

Sources (pinned revisions, decisions of 2026-09-25, see memory/lora-training-datasets):
  tasksource/procedural-typed-decisions b4f55be  all 11 configs
  ZefanCai/Open-Jev release-v2-redistributable c67699e  minus customer-control-v1 (TypeSafe doc
      wording, license unverified) and workflow-controls-v1 (same workflows as typed-decisions)
  Praveenrajus/jev-bench b41e6b2  configs with <= 26 options and a license allowing training
Evaluation sets are never read for training; every training state is checked against them.

  .venv-train/Scripts/python training/build_data.py            # writes .research/train-data/{train,dev}.jsonl
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, ROOT, Progress, log, read_jsonl, tokenizer

RAW = DATA_DIR / "raw"
MAX_TOKENS = 2048  # longer prompts are dropped, never truncated
SEED = 0
SMOOTH = 0.03  # label smoothing for hard human labels only (jev-bench); exact labels stay exact

TASKSOURCE = [
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
JEVBENCH = [
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
OPENJEV_EXCLUDED = ("customer-control-v1", "workflow-controls-v1")
# Questions per group and split. Open-Jev is capped per source, not per config: it is 30% pixel
# painting and 30% games, which matter less here than workflow-like decisions.
CAP = {"tasksource": (1400, 120), "jevbench": (800, 80), "openjev": (700, 60)}


def wire_to_native(state, questions):
    """compat.to_native without the wire limits (tasksource has 11-level scores); the native
    schema still validates everything."""
    from rizzo_flow import compat

    built = {}
    for key, q in questions.items():
        if q["type"] == "noul":
            criteria = q.get("criteria")
            built[key] = compat.NoulQuestion.model_construct(
                type="noul",
                instructions=q["instructions"],
                criteria=compat.NoulCriteria.model_construct(**criteria) if criteria else None,
            )
        elif q["type"] == "choice":
            built[key] = compat.ChoiceQuestion.model_construct(**q)
        else:
            built[key] = compat.ScoreQuestion.model_construct(**q)
    request = compat.SystemOneRequest.model_construct(state=state, model="x", questions=built)
    native, _ = compat.to_native(request)
    return native


def order(question):
    """Answer keys in slot order: what compat.to_native gives the letters A, B, C..."""
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def smooth(one_hot):
    k = len(one_hot)
    return [(1 - SMOOTH) * p + SMOOTH / k for p in one_hot]


def tasksource_rows(split):
    import pyarrow.parquet as pq

    for config in TASKSOURCE:
        for row in pq.read_table(RAW / "tasksource" / f"{config}__{split}.parquet").to_pylist():
            state = row["state"]
            try:
                parsed = json.loads(state)
                state = parsed if isinstance(parsed, (dict, list)) else state
            except json.JSONDecodeError:
                pass
            questions, answers = json.loads(row["questions"]), json.loads(row["answers"])
            targets = {}
            for key, q in questions.items():
                a = answers[key]
                if q["type"] == "noul":
                    targets[key] = [1 - a["noul"], a["noul"]]
                else:
                    targets[key] = [a["probabilities"][k] for k in order(q)]
            yield f"tasksource/{config}", row["id"], state, questions, targets


def jevbench_rows(split):
    for config in JEVBENCH:
        path = RAW / "jevbench" / f"{config}__{split}.jsonl"
        for row in read_jsonl(path):
            q = json.loads(row["question"])
            keys = order(q)
            soft = row["soft_label"]
            if isinstance(soft, str):
                soft = json.loads(soft)
            if soft is None:
                target = smooth(
                    [
                        float(
                            k == str(row["label"])
                            or (
                                q["type"] == "noul"
                                and k == ("true" if row["label"] == "1" else "false")
                            )
                        )
                        for k in keys
                    ]
                )
            elif q["type"] == "noul":
                target = [1 - float(soft), float(soft)]
            elif isinstance(soft, list):
                target = [float(v) for v in soft]
            else:
                target = [float(soft.get(k, 0.0)) for k in keys]
            yield f"jevbench/{config}", row["id"], json.loads(row["state"]), {"q": q}, {"q": target}


def openjev_rows(split):
    import pyarrow.parquet as pq

    for row in pq.read_table(RAW / "openjev" / f"{split}.parquet").to_pylist():
        source = row["source"].split("/")[0]
        if source in OPENJEV_EXCLUDED:
            continue
        options = list(row["options"])
        if row["kind"] == "noul":
            q = {"type": "noul", "instructions": row["question"]}
            p_yes = row["target"][options.index("yes")]
            target = [1 - p_yes, p_yes]
        elif row["kind"] == "choice":
            if len(set(options)) != len(options):
                continue
            q = {
                "type": "choice",
                "instructions": row["question"],
                "criteria": dict.fromkeys(options),
            }
            target = list(row["target"])
        else:
            q = {"type": "score", "instructions": row["question"], "criteria": options}
            target = list(row["target"])
        yield f"openjev/{source}", row["id"], json.loads(row["state_json"]), {"q": q}, {"q": target}


def eval_texts():
    """Long strings from every evaluation set we report on; none may appear in a training state."""
    texts = set()

    def walk(value):
        if isinstance(value, str):
            if len(value) >= 60:
                texts.add(value.strip())
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    files = [
        ROOT / ".research/typed-decisions/all/test.jsonl",
        ROOT / "benchmarks/smoke.jsonl",
        ROOT / "benchmarks/perturbations.jsonl",
        *(ROOT / ".research/SemIf/benchmarks/data").glob("*.jsonl"),
    ]
    missing = [p for p in files if not p.exists()]
    if missing or len(files) < 6:  # 3 own files + authored144, perturbations108, shape777
        raise SystemExit(
            f"Evaluation sets missing ({missing or 'SemIf fixtures'}): run training/download_data.py. "
            "Without them the contamination check would silently pass."
        )
    for path in files:
        for row in read_jsonl(path):
            for key in ("state", "request", "evidence", "provenance"):
                if key in row:
                    value = row[key]
                    walk(
                        json.loads(value) if isinstance(value, str) and value[:1] in "{[" else value
                    )
    return texts, [str(p.relative_to(ROOT)) for p in files]


def build(split, tok, forbidden):
    from rizzo_flow.prompts import compile_request
    from rizzo_flow.schema import Request

    rng = random.Random(f"{SEED}:{split}")
    by_group = defaultdict(list)
    for reader in (tasksource_rows, jevbench_rows, openjev_rows):
        for item in reader(split):
            by_group[item[0]].append(item)
    picked = []
    for group, items in sorted(by_group.items()):
        rng.shuffle(items)
        cap = CAP[group.split("/")[0]][0 if split == "train" else 1]
        count = 0
        for item in items:
            if count >= cap:
                break
            picked.append(item)
            count += len(item[3])
    log(f"{split}: {len(picked)} states from {len(by_group)} groups, compiling")
    stats = Counter()
    out = []
    progress = Progress(len(picked), f"compile {split}", every=500)
    for step, (group, rid, state, questions, targets) in enumerate(picked, 1):
        progress(step)
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if any(t in text for t in forbidden):
            stats["contaminated"] += 1
            continue
        try:
            native = Request.model_validate(
                wire_to_native(state, questions).model_dump(mode="json", exclude_none=True)
            )
            _, compiled = compile_request(tok, native, MAX_TOKENS)
        except ValueError as error:
            stats["too_long" if "context limit" in str(error) else "invalid"] += 1
            continue
        for job in compiled:
            target = targets[job.id]
            if len(target) != len(job.slots) or abs(sum(target) - 1) > 1e-3:
                stats["bad_target"] += 1
                continue
            out.append(
                {
                    "group": group,
                    "id": f"{rid}/{job.id}",
                    "tokens": job.tokens,
                    "slots": job.slots,
                    "target": target,
                }
            )
            stats[group] += 1
    rng.shuffle(out)
    return out, stats


def main():
    tok = tokenizer()
    forbidden, eval_files = eval_texts()
    log(
        f"{len(forbidden)} evaluation strings from {len(eval_files)} files are excluded from training"
    )
    manifest = {
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "smooth": SMOOTH,
        "caps": CAP,
        "eval_files": eval_files,
    }
    for split, name in (("train", "train"), ("validation", "dev")):
        rows, stats = build(split, tok, forbidden)
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "x", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
        lengths = sorted(len(r["tokens"]) for r in rows)
        manifest[name] = {
            "rows": len(rows),
            "tokens_p50": lengths[len(lengths) // 2],
            "tokens_max": lengths[-1],
            "tokens_total": sum(lengths),
            "stats": dict(sorted(stats.items())),
        }
        log(
            f"{name}: {len(rows)} rows, p50 {lengths[len(lengths) // 2]} tokens, dropped { ({k: v for k, v in stats.items() if '/' not in k}) }"
        )
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
