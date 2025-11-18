#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
from datetime import datetime

HEADER = "timestamp_utc_iso,category,input_file,n,seed,rep_id,elapsed_ms,cpu_ms,comparisons,swaps,correct"


def find_executable(project_root: str) -> str:
    candidates = [
        os.path.join(project_root, "build", "random_qsort"),
        os.path.join(project_root, "random_qsort"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    which = shutil.which("random_qsort")
    if which:
        return which
    return ""


def list_categories(dataset_root: str):
    cats = []
    if not os.path.isdir(dataset_root):
        return cats
    for entry in os.scandir(dataset_root):
        if entry.is_dir():
            cats.append(entry.name)
    cats.sort()
    return cats


def list_txt_files_sorted_by_size(folder: str):
    files = []
    if not os.path.isdir(folder):
        return files
    for entry in os.scandir(folder):
        if entry.is_file() and entry.name.endswith('.txt'):
            files.append(entry.path)
    files.sort(key=lambda p: os.path.getsize(p))
    return files


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_iso():
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def run():
    parser = argparse.ArgumentParser(description="Run random_qsort over all datasets and aggregate results.")
    parser.add_argument("--dataset-root", default="./datasets", help="Root folder containing category subfolders")
    parser.add_argument("--out-root", default="./results/qsort", help="Base output folder")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed (default 42)")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per file (default 5)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    exe = find_executable(project_root)
    if not exe:
        print("Error: Could not find 'random_qsort' executable. Build it first (see README).", file=sys.stderr)
        sys.exit(2)

    ensure_dir(args.out_root)

    master_csv = os.path.join(args.out_root, "qsort_master.csv")
    failures_log = os.path.join(args.out_root, "failures.log")

    # Write master header if missing/empty
    need_master_header = (not os.path.exists(master_csv)) or (os.path.getsize(master_csv) == 0)
    if need_master_header:
        with open(master_csv, 'w') as f:
            f.write(HEADER + "\n")

    categories = list_categories(args.dataset_root)
    if not categories:
        print(f"No categories found under {args.dataset_root}. Nothing to do.")
        sys.exit(0)

    total_runs = 0
    successes = 0
    failures = 0

    for cat in categories:
        cat_folder = os.path.join(args.dataset_root, cat)
        files = list_txt_files_sorted_by_size(cat_folder)
        if not files:
            print(f"[skip] Category '{cat}' has no .txt files.")
            continue

        cat_csv = os.path.join(args.out_root, f"{cat}.csv")
        # Ensure per-category CSV has header if empty
        if (not os.path.exists(cat_csv)) or (os.path.getsize(cat_csv) == 0):
            with open(cat_csv, 'w') as f:
                f.write(HEADER + "\n")

        print(f"Processing category '{cat}' with {len(files)} files…")

        for file_idx, fpath in enumerate(files):
            for rep in range(args.reps):
                seed = int(args.seed_base + file_idx * 1000 + rep)
                cmd = [
                    exe,
                    "--input-file", fpath,
                    "--seed", str(seed),
                    "--rep", str(rep),
                    "--out-csv", cat_csv,
                ]
                total_runs += 1
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                except Exception as e:
                    failures += 1
                    with open(failures_log, 'a') as flog:
                        flog.write(f"{now_iso()} | EXCEPTION | cat={cat} file={fpath} rep={rep} seed={seed} | {e}\n")
                    print(f"[fail] {cat} {os.path.basename(fpath)} rep={rep} (exception)")
                    continue

                if res.returncode != 0:
                    failures += 1
                    with open(failures_log, 'a') as flog:
                        msg = res.stderr.strip().replace('\n', ' ')
                        flog.write(f"{now_iso()} | RC={res.returncode} | cat={cat} file={fpath} rep={rep} seed={seed} | {msg}\n")
                    print(f"[fail] {cat} {os.path.basename(fpath)} rep={rep} rc={res.returncode}")
                    continue

                # On success, pick the last non-empty stdout line as the row (skip header if any)
                out_lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                row_line = ""
                for ln in reversed(out_lines):
                    if not ln.startswith("timestamp_utc_iso"):
                        row_line = ln
                        break
                if not row_line:
                    # Fallback: read the last line from the category CSV
                    try:
                        with open(cat_csv, 'r') as cf:
                            lines = [l.strip() for l in cf if l.strip()]
                        if len(lines) >= 2:
                            row_line = lines[-1]
                    except Exception:
                        pass

                if row_line:
                    with open(master_csv, 'a') as mf:
                        mf.write(row_line + "\n")
                    successes += 1
                    if successes % 20 == 0:
                        print(f"  progress: {successes} successes / {total_runs} runs")
                else:
                    failures += 1
                    with open(failures_log, 'a') as flog:
                        flog.write(f"{now_iso()} | WARN | cat={cat} file={fpath} rep={rep} seed={seed} | No row captured from stdout or cat CSV.\n")
                    print(f"[warn] No row captured for {cat} {os.path.basename(fpath)} rep={rep}")

    print("\nSummary:")
    print(f"  total runs   : {total_runs}")
    print(f"  successes    : {successes}")
    print(f"  failures     : {failures}")


if __name__ == "__main__":
    run()
