"""Step 0 (SPEC v1 §0, §9.2): verify the OSF snapshot and de-identify the behavioral files.

Reads raw files from RAW (outside the repository), checks each SHA-256 against
RAW/MANIFEST.tsv, and writes de-identified behavioral files to data/ (gitignored).

De-identification:
- worker_id and participant_id are replaced by a random code (H/C/S prefix + 3 digits)
  drawn once with `secrets` and stored only in RAW/deid_key.csv, never in the repository.
  A worker appearing in more than one file keeps one code (prefix W).
- device_type, device_os, device_browser and all wall-clock timestamps are dropped.
  Order within participant is kept as `task_order` (rank of task_end_time) and
  `t_end_s` (seconds since that participant's first task_end_time).
- the database row id and the pandas index column are dropped; `row` is a new key.
Prediction files carry no personal data and are read from RAW directly; they are not copied.
"""
import csv
import hashlib
import secrets
from pathlib import Path

import pandas as pd

RAW = Path.home() / "Desktop" / "hai-reference-raw"
OUT = Path(__file__).resolve().parents[1] / "data"

BEHAVIORAL = {
    "h16": ("2ntrf/human_only_classification_6per_img_preprocessed.csv", "H"),
    "tejeda_concurrent": ("9xbpu/data_concurrent_paradigm.csv", "C"),
    "tejeda_sequential": ("9xbpu/data_sequential_paradigm.csv", "S"),
}
DROP = ["Unnamed: 0", "id", "worker_id", "participant_id", "device_type", "device_os",
        "device_browser", "experiment_start_time", "experiment_end_time",
        "task_start_time", "task_end_time"]


def verify_snapshot():
    with open(RAW / "MANIFEST.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        p = RAW / r["project"] / Path(r["osf_path"]).name
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != r["osf_sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {p}")
    return len(rows)


def load_key(frames):
    key_path = RAW / "deid_key.csv"
    if key_path.exists():
        return pd.read_csv(key_path)
    files_per_worker = {}
    for name, d in frames.items():
        for w in d.worker_id.unique():
            files_per_worker.setdefault(w, []).append(name)
    rows, used = [], set()
    for name, (_, prefix) in BEHAVIORAL.items():
        workers = [w for w in frames[name].worker_id.unique() if w not in {r["worker_id"] for r in rows}]
        secrets.SystemRandom().shuffle(workers)
        for w in workers:
            pre = "W" if len(files_per_worker[w]) > 1 else prefix
            while True:
                code = f"{pre}{secrets.randbelow(900) + 100}"
                if code not in used:
                    break
            used.add(code)
            rows.append({"worker_id": w, "code": code})
    key = pd.DataFrame(rows)
    key.to_csv(key_path, index=False)
    return key


def deidentify(d, key):
    d = d.merge(key, on="worker_id", how="left", validate="many_to_one")
    assert d.code.notna().all()
    assert d.groupby("code").participant_id.nunique().max() == 1
    end = pd.to_datetime(d.task_end_time, format="ISO8601")
    d["task_order"] = end.groupby(d.code).rank(method="first").astype(int)
    d["t_end_s"] = (end - end.groupby(d.code).transform("min")).dt.total_seconds().round(3)
    d = d.drop(columns=[c for c in DROP if c in d.columns])
    d = d.sort_values(["code", "task_order"]).reset_index(drop=True)
    d.insert(0, "row", range(1, len(d) + 1))
    cols = ["row", "code", "task_order", "t_end_s"]
    return d[cols + [c for c in d.columns if c not in cols]]


def main():
    n = verify_snapshot()
    frames = {k: pd.read_csv(RAW / f, low_memory=False) for k, (f, _) in BEHAVIORAL.items()}
    key = load_key(frames)
    OUT.mkdir(exist_ok=True)
    raw_ids = set(key.worker_id)
    for name, d in frames.items():
        out = deidentify(d, key)
        text = out.to_csv(index=False)
        assert not any(w in text for w in raw_ids), f"worker id leaked into {name}"
        (OUT / f"{name}_deid.csv").write_text(text)
        print(f"{name}: {len(d)} rows -> {len(out)} rows, {out.code.nunique()} codes")
    print(f"snapshot verified: {n} files; workers in more than one file: {(key.code.str[0] == 'W').sum()}")


if __name__ == "__main__":
    main()
