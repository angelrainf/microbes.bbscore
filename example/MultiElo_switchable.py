#!/usr/bin/env python3

"""
Multiplayer Elo — classic vs corrected (switchable)

Modes:
- classic   : no absence correction (standard multiplayer Elo)
- corrected : apply absence correction by propagating the last-place delta
             to all absentees after each game

CLI:
  --mode classic|corrected
  --input <dir or glob>                 (default: MatrixForElo)
  --output <dir>                        (default: ../02_results_Elo_Python)
  --subsample <int>                     (#samples/iter; if <=0 or ≥total → ALL)
  --iters <int>                         (default 1)
  --workers <int>                       (default 4)
  --seed <int>                          (default 666)
  --K <float>                           (default 10)
  --D <float>                           (default 400)
  --coef-file <path.csv>                (CSV with biome-specific base coef)
  --coef-value <float>                  (fixed base coef for all biomes; overrides file)

If both --coef-file and --coef-value are provided, the fixed --coef-value takes precedence.
"""

import argparse
import csv
import glob
import os
import random
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Defaults (overridable by CLI)
SUMMARY_CSV_DEFAULT = "MAP_summary_biomes_samples.csv"   # used only if --coef-file not provided
DEFAULT_INPUT_DIR = "MatrixForElo"
DEFAULT_OUTPUT_DIR = "../02_results_Elo_Python"
DEFAULT_PATTERN = "01percentfilter_Matrix_MAPbiomes_*.csv"

# Column names in the summary CSV (when using --coef-file)
BIOME_COLUMN = "MAP_biomes"
COEF_COLUMN = "Elo.coef"


# ---------- helpers ----------
def resolve_input_glob(input_arg: str, default_pattern: str = DEFAULT_PATTERN) -> str:
    """If input_arg is a directory, append default_pattern. Otherwise treat input_arg as a glob pattern."""
    if os.path.isdir(input_arg):
        return os.path.join(input_arg, default_pattern)
    return input_arg


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------- absence-correction helpers (used only in corrected mode) ----------
def compute_bottom_delta(old_ratings: Dict[str, float],
                         updated_ratings: Dict[str, float],
                         present_order: List[str]) -> float:
    """Delta for last-ranked present OTU; applied to absentees in corrected mode."""
    last = present_order[-1]
    return updated_ratings[last] - old_ratings[last]


def apply_absence_correction(ratings: Dict[str, float],
                             all_ids: Set[str],
                             present_order: List[str],
                             delta: float) -> Dict[str, float]:
    """Add delta to all OTUs absent from this game."""
    absent = all_ids - set(present_order)
    for pid in absent:
        ratings[pid] += delta
    return ratings


# ---------- core logic ----------
def load_games_matrix(path: str) -> Tuple[List[List[str]], Set[str]]:
    """
    Load matrix, skipping empty/None/nan player cells.
    Returns list of games (ordered OTUs) and set of all valid OTU IDs.
    """
    df = pd.read_csv(path, keep_default_na=False, na_values=["NaN", "None"], dtype=str)
    cols = [c for c in df.columns if c.lower() != "date"]
    games: List[List[str]] = []
    all_players: Set[str] = set()
    for _, row in df.iterrows():
        present: List[str] = []
        for c in cols:
            cell = row[c]
            if pd.isna(cell):
                continue
            cell = str(cell).strip().strip('"')
            if not cell or cell.lower() in {"none", "nan"}:
                continue
            present.append(cell)
            all_players.add(cell)
        games.append(present)
    all_players = {pid for pid in all_players if pid and pid.lower() not in {"none", "nan"}}
    return games, all_players


def initialize_ratings(player_ids: Set[str]) -> Dict[str, float]:
    return {pid: 1000.0 for pid in player_ids}


def compute_actual_scores(
    present_order: List[str],
    base_coef: float,
    result_order: Optional[List[int]] = None
) -> List[float]:
    """
    Exponential scorer with tie handling. Returns scores summing to 1.
    """
    n = len(present_order)
    if result_order is None:
        result_order = list(range(1, n + 1))  # 1..n

    raw = np.array([(base_coef ** (n - r) - 1) for r in result_order], dtype=float)

    # tie averaging
    for rank in set(result_order):
        idxs = [i for i, r in enumerate(result_order) if r == rank]
        if len(idxs) > 1:
            raw[idxs] = raw[idxs].mean()

    total = raw.sum()
    return (raw / total).tolist() if total > 0 else [0.0] * n


def update_present_ratings(
    ratings: Dict[str, float],
    present_order: List[str],
    base_coef: float,
    result_order: Optional[List[int]] = None,
    k: float = 10.0,
    d: float = 400.0
) -> Dict[str, float]:
    """
    Compute new ratings for the players in present_order.
    Expected scores normalized so sum = 1; supports ties in result_order.
    """
    n = len(present_order)
    actual_scores = compute_actual_scores(present_order, base_coef, result_order)

    # ratings vector for present players
    R = np.array([ratings[pid] for pid in present_order], dtype=float)

    # pairwise win probabilities (base-10 logistic)
    diff = R[:, None] - R[None, :]
    P = 1.0 / (1.0 + 10.0 ** (-diff / d))
    np.fill_diagonal(P, 0.0)

    # expected score for each player, normalized to sum=1
    exp_sum = P.sum(axis=1)
    denom = n * (n - 1) / 2
    expected_scores = (exp_sum / denom).tolist()

    # Elo update; (by design, no extra *(n-1) scaling)
    scale = k
    return {
        pid: ratings[pid] + scale * (S - E)
        for pid, S, E in zip(present_order, actual_scores, expected_scores)
    }


def simulate_games(games: List[List[str]],
                   all_ids: Set[str],
                   base_coef: float,
                   mode_corrected: bool,
                   k: float,
                   d: float) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Run through a list of games and update ratings in sequence.
    If mode_corrected is True, apply absence correction each game.
    """
    ratings = initialize_ratings(all_ids)
    counts = {pid: 0 for pid in all_ids}
    for present in games:
        for pid in present:
            counts[pid] += 1

        if mode_corrected:
            old = ratings.copy()
            updated = update_present_ratings(ratings, present, base_coef, k=k, d=d)
            delta = compute_bottom_delta(old, updated, present)
            ratings.update(updated)
            ratings = apply_absence_correction(ratings, all_ids, present, delta)
        else:
            updated = update_present_ratings(ratings, present, base_coef, k=k, d=d)
            ratings.update(updated)

    return ratings, counts


def run_iteration(it: int,
                  games_matrix: List[List[str]],
                  all_player_ids: Set[str],
                  total_games: int,
                  biome: str,
                  base_coef: float,
                  mode_corrected: bool,
                  subsample_size: Optional[int],
                  seed: int,
                  k: float,
                  d: float) -> pd.DataFrame:
    """
    One iteration: choose subsample (or all), shuffle order deterministically, run Elo.
    """
    rnd = random.Random(seed + it)

    # choose indices
    if (subsample_size is None) or (subsample_size <= 0) or (subsample_size >= total_games):
        idxs = list(range(total_games))  # ALL samples
        rnd.shuffle(idxs)
    else:
        idxs = rnd.sample(range(total_games), subsample_size)

    sampled = [games_matrix[i] for i in idxs]

    ratings, counts = simulate_games(sampled, all_player_ids, base_coef, mode_corrected, k, d)
    df = pd.DataFrame({
        'player_id': list(ratings.keys()),
        'n_games': [counts[pid] for pid in ratings.keys()],
        'rating': list(ratings.values())
    })
    df = df.sort_values('rating', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['run'] = it
    df['biome'] = biome
    df['mode'] = 'corrected' if mode_corrected else 'classic'
    df['K'] = k
    df['D'] = d
    df['base_coef'] = base_coef
    return df[['rank', 'player_id', 'n_games', 'rating', 'run', 'biome']]


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Multiplayer Elo with optional absence correction and flexible base coefficient source.")
    parser.add_argument("--mode", choices=["classic", "corrected"], default="classic",
                        help="Choose 'classic' (no absence correction) or 'corrected' (apply absence correction).")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR,
                        help=f"Directory containing CSVs OR a glob pattern. If a directory is provided, "
                             f"files matching '{DEFAULT_PATTERN}' inside it will be used.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for result CSVs (will be created if missing).")
    parser.add_argument("--subsample", type=int, default=None,
                        help="Number of samples per iteration. If omitted/≤0/≥total, uses ALL samples.")
    parser.add_argument("--iters", type=int, default=1,
                        help="Number of iterations to run (default 1).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for iterations (default 4).")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed for reproducibility (default 666).")
    parser.add_argument("--K", type=float, default=10.0,
                        help="Elo K-factor (default 10).")
    parser.add_argument("--D", type=float, default=400.0,
                        help="Elo D scale (default 400).")
    parser.add_argument("--coef-file", type=str, default=None,
                        help=f"CSV file with per-biome base coefficients (default columns: {BIOME_COLUMN}, {COEF_COLUMN}). "
                             f"If not provided, falls back to {SUMMARY_CSV_DEFAULT} if it exists.")
    parser.add_argument("--coef-value", type=float, default=None,
                        help="Fixed base coefficient value to use for ALL biomes. Overrides --coef-file if provided.")
    args = parser.parse_args()

    mode_corrected = (args.mode == "corrected")
    input_glob = resolve_input_glob(args.input)
    output_dir = args.output
    ensure_dir(output_dir)

    # Global seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # K and D
    k_val = float(args.K)
    d_val = float(args.D)

    # Determine base coefficient source
    fixed_coef: Optional[float] = args.coef_value
    coef_file: Optional[str] = args.coef_file

    base_coef_map: Optional[Dict[str, float]] = None
    using_fixed = fixed_coef is not None

    if not using_fixed:
        # If no explicit file provided, try the default summary CSV if it exists
        if coef_file is None and os.path.isfile(SUMMARY_CSV_DEFAULT):
            coef_file = SUMMARY_CSV_DEFAULT

        if coef_file is not None:
            if not os.path.isfile(coef_file):
                raise FileNotFoundError(
                    f"Coefficient file '{coef_file}' not found and no --coef-value provided."
                )
            summary = pd.read_csv(coef_file, keep_default_na=False, na_values=["NaN"]).replace({"None": None})
            if BIOME_COLUMN not in summary.columns or COEF_COLUMN not in summary.columns:
                raise ValueError(
                    f"Coefficient file must contain columns '{BIOME_COLUMN}' and '{COEF_COLUMN}'."
                )
            base_coef_map = summary.set_index(BIOME_COLUMN)[COEF_COLUMN].astype(float).to_dict()
        else:
            raise FileNotFoundError(
                f"No coefficient file provided and default '{SUMMARY_CSV_DEFAULT}' not found. "
                f"Provide --coef-file or a fixed --coef-value."
            )

    n_iter = max(1, int(args.iters))  # ensure ≥1
    subs_desc = "ALL" if args.subsample is None or args.subsample <= 0 else str(args.subsample)

    for filepath in glob.glob(input_glob):
        base = os.path.basename(filepath)
        name = base.replace("MatrixRank_MAPbiomes_", "").replace(".csv", "")

        # Choose base_coef
        if using_fixed:
            base_coef = float(fixed_coef)
        else:
            if name not in base_coef_map:
                print(f"⚠️ Skipping {name} (no base_coef found in {coef_file})")
                continue
            base_coef = float(base_coef_map[name])

        suffix = "Corrected" if mode_corrected else "Classic"
        coef_tag = f"coef{base_coef:.4g}" if using_fixed else "coefFile"
        out_file = os.path.join(
            output_dir,
            f"MultiElo_{suffix}_sub{subs_desc}_iters{n_iter}_K{k_val}_D{d_val}_{coef_tag}_{name}.csv"
        )

        print(f"Processing {base} → {out_file} "
              f"(mode={args.mode}, base_coef={'fixed' if using_fixed else 'file'}, value={base_coef:.6g}, "
              f"subsample={subs_desc}, iters={n_iter}, workers={args.workers}, seed={args.seed}, K={k_val}, D={d_val})")

        games_matrix, all_player_ids = load_games_matrix(filepath)
        total_games = len(games_matrix)
        results: List[pd.DataFrame] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_iteration,
                    i,
                    games_matrix,
                    all_player_ids,
                    total_games,
                    name,
                    base_coef,
                    mode_corrected,
                    args.subsample,
                    args.seed,
                    k_val,
                    d_val,
                )
                for i in range(1, n_iter + 1)
            ]
            for idx, fut in enumerate(as_completed(futures), start=1):
                results.append(fut.result())
                if idx % 50 == 0 or idx == n_iter:
                    print(f"  Completed {idx}/{n_iter} runs for {name}...")

        df_all = pd.concat(results, ignore_index=True)
        df_all.to_csv(out_file, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
        print(f"Finished {name}; saved to {out_file}\n")


if __name__ == "__main__":
    main()
