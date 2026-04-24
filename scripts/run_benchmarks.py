#!/usr/bin/env python3
"""
run_benchmarks.py
──────────────────────────────────────────────────────────────────────────────
Orchestration script: extracts .tar.gz archives, converts each MTX to binary
CSR, runs row_spgemm and tile_spgemm on each matrix, collects JSON results,
and writes a combined results.json to the results/ directory.

Called by the Makefile after the binaries are built.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import glob
import json
import subprocess
import tarfile
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
DATASET_DIR  = os.path.join(PROJECT_DIR, 'Dataset')
CSR_DIR      = os.path.join(PROJECT_DIR, 'csr_cache')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results')
GRAPHS_DIR   = os.path.join(PROJECT_DIR, 'graphs')
TILE_BIN     = os.path.join(PROJECT_DIR, 'build', 'tile_spgemm')
ROW_BIN      = os.path.join(PROJECT_DIR, 'build', 'row_spgemm')
PYTHON       = sys.executable

os.makedirs(CSR_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR,  exist_ok=True)

CONVERT_SCRIPT = os.path.join(SCRIPT_DIR, 'convert_mtx.py')

def run(cmd, check=True):
    """Run a shell command, stream stderr to terminal."""
    result = subprocess.run(cmd, capture_output=False, text=True,
                             stderr=sys.stderr, stdout=subprocess.PIPE)
    if check and result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}", file=sys.stderr)
    return result

def _extract_archives():
    """Extract all dataset archives present at the Dataset root."""
    archive_patterns = ('*.tar.gz', '*.tgz')
    archives = []
    for pattern in archive_patterns:
        archives.extend(glob.glob(os.path.join(DATASET_DIR, pattern)))

    for arch in sorted(set(archives)):
        stem = os.path.basename(arch)
        if stem.endswith('.tar.gz'):
            name = stem[:-7]
        elif stem.endswith('.tgz'):
            name = stem[:-4]
        else:
            continue
        extract_dir = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(extract_dir):
            print(f"\n[Prepare] Extracting {os.path.basename(arch)} ...", flush=True)
            with tarfile.open(arch, 'r:gz') as tf:
                tf.extractall(DATASET_DIR)


def find_mtx_files():
    """Return all datasets currently available under Dataset/."""
    _extract_archives()

    dataset_root = Path(DATASET_DIR)
    mtx_paths = sorted(dataset_root.rglob('*.mtx'))
    if not mtx_paths:
        return []

    discovered = {}
    for mtx in mtx_paths:
        rel_parts = mtx.relative_to(dataset_root).parts
        if len(rel_parts) > 1:
            name = rel_parts[0]
        else:
            name = mtx.stem

        # Prefer Dataset/<name>/<name>.mtx when multiple MTX files exist.
        current = discovered.get(name)
        preferred = (mtx.stem == name)
        if current is None:
            discovered[name] = str(mtx)
        else:
            current_path = Path(current)
            current_preferred = (current_path.stem == name)
            if preferred and not current_preferred:
                discovered[name] = str(mtx)

    return sorted(discovered.items())

def convert_to_csr(name, mtx_path):
    csr_path = os.path.join(CSR_DIR, name + '.csr')
    if os.path.exists(csr_path):
        csr_mtime = os.path.getmtime(csr_path)
        src_mtime = os.path.getmtime(mtx_path)
        if csr_mtime >= src_mtime:
            print(f"  [convert] {name}.csr is up to date, skipping.", flush=True)
            return csr_path
        print(f"  [convert] {name}.csr is stale, rebuilding from {mtx_path}.", flush=True)
    print(f"\n[Convert] {name}: {mtx_path} → {csr_path}", flush=True)
    result = run([PYTHON, CONVERT_SCRIPT, mtx_path, csr_path])
    if result.returncode != 0:
        return None
    return csr_path

def run_tile_spgemm(name, csr_path):
    save_c = os.path.join(CSR_DIR, name + '_C_tile.bin')
    print(f"\n[TileSpGEMM] Running on matrix: {name}", flush=True)
    print(f"  Phase: format conversion + Step1(tile structure) + Step2(symbolic) + Step3(numeric)", flush=True)
    result = run([TILE_BIN, csr_path, name, '--save-c', save_c])
    if result.returncode != 0 or not result.stdout.strip():
        print(f"  [WARN] TileSpGEMM failed or no output for {name}", file=sys.stderr)
        return None, save_c
    line = result.stdout.strip().split('\n')[-1]
    try:
        data = json.loads(line)
    except:
        print(f"  [WARN] Could not parse JSON: {line}", file=sys.stderr)
        return None, save_c
    return data, save_c

def run_row_spgemm(name, csr_path, tile_c_path):
    print(f"\n[RowSpGEMM] Running on matrix: {name}", flush=True)
    print(f"  Phase: upload to GPU + cuSPARSE SpGEMM", flush=True)
    check_args = ['--check', tile_c_path] if os.path.exists(tile_c_path) else []
    cmd = [ROW_BIN, csr_path, name] + check_args
    result = run(cmd)
    if result.returncode != 0 or not result.stdout.strip():
        print(f"  [WARN] RowSpGEMM failed or no output for {name}", file=sys.stderr)
        return None
    line = result.stdout.strip().split('\n')[-1]
    try:
        data = json.loads(line)
    except:
        print(f"  [WARN] Could not parse JSON: {line}", file=sys.stderr)
        return None
    return data

def main():
    print("="*70, flush=True)
    print("  SpGEMM Benchmark Suite — TileSpGEMM vs RowSpGEMM", flush=True)
    print("="*70, flush=True)

    # Check binaries
    for b in [TILE_BIN, ROW_BIN]:
        if not os.path.exists(b):
            print(f"[ERROR] Binary not found: {b}\nRun 'make build' first.", file=sys.stderr)
            sys.exit(1)

    mtx_files = find_mtx_files()
    if not mtx_files:
        print("[ERROR] No .mtx files found in Dataset/", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Info] Found {len(mtx_files)} matrices: {[n for n,_ in mtx_files]}", flush=True)

    all_results = []

    for name, mtx_path in mtx_files:
        print(f"\n{'─'*60}", flush=True)
        print(f"  Processing matrix: {name}", flush=True)
        print(f"{'─'*60}", flush=True)

        # Step 1: Convert
        csr_path = convert_to_csr(name, mtx_path)
        if not csr_path:
            continue

        # Step 2: TileSpGEMM
        tile_data, tile_c_path = run_tile_spgemm(name, csr_path)

        # Step 3: RowSpGEMM (+ correctness check against tile result)
        row_data = run_row_spgemm(name, csr_path, tile_c_path)

        entry = {'matrix': name}
        if tile_data:
            entry['tile'] = tile_data
            print(f"  [Result] TileSpGEMM: {tile_data.get('time_ms',0):.2f} ms, "
                  f"{tile_data.get('gflops',0):.2f} GFlops", flush=True)
        if row_data:
            entry['row']  = row_data
            print(f"  [Result] RowSpGEMM:  {row_data.get('time_ms',0):.2f} ms, "
                  f"{row_data.get('gflops',0):.2f} GFlops", flush=True)

        all_results.append(entry)

    # Save combined results
    results_file = os.path.join(RESULTS_DIR, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved to {results_file}", flush=True)

if __name__ == '__main__':
    main()
