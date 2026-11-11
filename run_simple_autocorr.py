"""
Run centered autocorrelation analysis for protein fitness landscapes.

The calculation functions are imported from simple_autocorr_functions.py.

For every input landscape, the script writes five tables:
    <landscape>_hamming.csv
    <landscape>_atchley.csv
    <landscape>_cum_hamming.csv
    <landscape>_cum_atchley.csv
    <landscape>_summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd

import simple_autocorr_functions as af


def run_one(file, out_dir, max_h, max_a, threads, chunk_size):
    """
    Process one landscape and save exact and cumulative result tables.

    input:
        file - path to one landscape file
        out_dir - output directory
        max_h - maximum Hamming distance
        max_a - maximum Atchley distance
        threads - number of parallel worker processes
        chunk_size - number of outer-loop indices i in one chunk

    output:
        The function saves Hamming, Atchley, cumulative, and summary tables.
    """

    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = file.stem

    data, wt, info = af.load_data(file)

    # Chunk size is used here, outside corr().
    tasks = af.make_tasks(len(data), chunk_size)

    print("\n" + "=" * 70)
    print(f"Landscape: {file}")
    print(f"Rows: {info['rows']:,}")
    print(f"Unique genotypes: {info['n']:,}")
    print(f"Duplicates skipped: {info['duplicates']:,}")
    n = info["n"]
    print(f"Unique pairs including self-pairs: {n * (n + 1) // 2:,}")
    print(f"Threads: {threads}")
    print(f"Chunk size: {chunk_size}")
    print(f"Number of chunks: {len(tasks):,}")

    h_df, a_df, stat = af.corr(
        data,
        tasks=tasks,
        max_h=max_h,
        max_a=max_a,
        threads=threads,
    )

    h_cum = af.make_cumulative_autocorr(h_df)
    a_cum = af.make_cumulative_autocorr(a_df)

    h_path = out_dir / f"{name}_hamming.csv"
    a_path = out_dir / f"{name}_atchley.csv"
    hc_path = out_dir / f"{name}_cum_hamming.csv"
    ac_path = out_dir / f"{name}_cum_atchley.csv"
    s_path = out_dir / f"{name}_summary.csv"

    h_df.to_csv(h_path, index=False)
    a_df.to_csv(a_path, index=False)
    h_cum.to_csv(hc_path, index=False)
    a_cum.to_csv(ac_path, index=False)

    summary = {
        "landscape": name,
        "input_file": str(file),
        "rows_in_file": info["rows"],
        "unique_genotypes": info["n"],
        "duplicates_skipped": info["duplicates"],
        **stat,
        "threads": threads,
        "chunk_size": chunk_size,
        "number_of_chunks": len(tasks),
        "wt_positions": len(wt),
        "max_hamming": max_h,
        "max_atchley": max_a,
        "atchley_binning": "ceil",
        "hamming_table": str(h_path),
        "atchley_table": str(a_path),
        "cumulative_hamming_table": str(hc_path),
        "cumulative_atchley_table": str(ac_path),
    }

    pd.DataFrame([summary]).to_csv(s_path, index=False)

    print(f"Saved: {h_path}")
    print(f"Saved: {a_path}")
    print(f"Saved: {hc_path}")
    print(f"Saved: {ac_path}")
    print(f"Saved: {s_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", default="autocorr_results")
    p.add_argument("--max-h", type=int, default=25)
    p.add_argument("--max-a", type=int, default=60)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--chunk-size", type=int, default=16)

    return p.parse_args()


def main():
    args = get_args()

    for file in args.files:
        run_one(
            file=file,
            out_dir=args.out,
            max_h=args.max_h,
            max_a=args.max_a,
            threads=args.threads,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
