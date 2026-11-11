"""
Fitness difference relative to the wild type:

    D_wt(d) = mean(|f_i - f_wt| at distance d from WT) / sigma_f

Keep simple_autocorr_functions.py in the same directory.

Every unique genotype is compared with WT once, including WT itself at d = 0.
Atchley distances are rounded up using ceil.
Output: wt_difference_hamming.csv and wt_difference_atchley.csv.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from simple_autocorr_functions import load_data, hdist, adist_g


def make_df(sums, counts, sigma):
    """
    Convert binned sums and genotype counts into a WT difference table.
    """
    rows = []
    for distance in sorted(counts):
        mean_abs = sums[distance] / counts[distance]
        rows.append([distance, counts[distance], mean_abs, mean_abs / sigma])

    return pd.DataFrame(
        rows,
        columns=["distance", "genotype_count", "mean_abs_difference_wt", "D_wt"],
    )


def wt_difference(data):
    """
    Compare each loaded genotype with the measured WT.

    data contains (genotype_dict, position_set, fitness) tuples from load_data.
    An empty genotype dictionary represents WT. Exactly one WT must be present.
    sigma is the population standard deviation (ddof=0) of the entire landscape,
    including WT, as in global_difference.py.

    Return Hamming and Atchley tables for individual distance bins.
    """

    wt_rows = [row for row in data if not row[0]]
    wt_g, wt_positions, wt_fitness = wt_rows[0]

    fitness = np.array([row[2] for row in data], dtype=float)
    sigma = np.std(fitness, ddof=0)

    h_sums, h_counts = defaultdict(float), defaultdict(int)
    a_sums, a_counts = defaultdict(float), defaultdict(int)

    # 1 comparison per genotype
    for genotype, positions, fit in data:
        difference = abs(fit - wt_fitness)

        dh = hdist(genotype, positions, wt_g, wt_positions)
        h_sums[dh] += difference
        h_counts[dh] += 1

        da = int(np.ceil(adist_g(genotype, positions, wt_g, wt_positions)))
        a_sums[da] += difference
        a_counts[da] += 1

    return make_df(h_sums, h_counts, sigma), make_df(a_sums, a_counts, sigma)


def run_one(input_file, output_dir):
    """
    Load the full landscape, calculate D_wt(d), and save two CSV tables.
    """
    data, _, _ = load_data(input_file)
    h_df, a_df = wt_difference(data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h_df.to_csv(output_dir / "wt_difference_hamming.csv", index=False)
    a_df.to_csv(output_dir / "wt_difference_atchley.csv", index=False)

    return h_df, a_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    run_one(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
