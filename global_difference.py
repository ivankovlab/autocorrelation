"""
Global fitness difference calculation:

    D(d) = mean(|f_i - f_j| at distance d) / sigma_f

Keep simple_autocorr_functions.py in the same directory. 

Output: two CSV files with distance, pair_count, mean_abs_difference, and D.

Atchley distances are rounded up using ceil.
Every unique pair is included once, including self-pairs.
"""

import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from simple_autocorr_functions import AA, load_data, hdist, adist_g, make_tasks


#### 1. Shared functions ####

# Data loading, distances, and chunk creation are imported above.
# They are the same functions used for autocorrelation.


#### 2. Difference table ####


def make_df(s1, count, sigma):
    """
    Convert pair statistics into a result table, as in autocorrelation.

    input:
        s1[d] - sum of absolute fitness differences at distance d;
        count[d] - number of pairs at distance d;
        sigma - fitness standard deviation of the entire landscape.

    output:
        table with four columns:
            distance - distance (ceil-bin index for Atchley);
            pair_count - number of unique pairs, including self-pairs at d = 0;
            mean_abs_difference - mean difference before normalization;
            D - mean difference divided by sigma.

    Formula:
        D(d) = s1[d] / count[d] / sigma.
    """
    rows = []

    for d in sorted(count):
        mean_abs = s1[d] / count[d]
        D = mean_abs / sigma
        rows.append([d, count[d], mean_abs, D])

    return pd.DataFrame(
        rows, columns=["distance", "pair_count", "mean_abs_difference", "D"]
    )


#### 3. Multiprocessing ####

_DATA = None


def init_worker(data):
    """
    Save the landscape inside each worker process once. The complete landscape does not have to be sent again with every task.

    input:
        data - list of (genotype_dict, position_set, fitness).

    output:
        None; data are saved in the global variable _DATA.
    """
    global _DATA
    _DATA = data


def work(task):
    """
    Calculate sums of differences and pair counts for one chunk.

    input:
        task = (i1, i2) - outer-loop indices i from i1 up to, but not including, i2.

    output:
        hs1, hc - sums and pair counts for Hamming distances;
        as1, ac - sums and pair counts for Atchley distance bins.

    For every i in the chunk, compare it with ALL j > i, including other chunks.
    Each unique pair (i, j) is added once; the reversed pair is not repeated.
    Each worker returns its own dictionaries; the main process combines them.
    """
    i1, i2 = task
    data = _DATA
    n = len(data)

    # new distances start with a sum and count of zero.
    hs1 = defaultdict(float)
    hc = defaultdict(int)
    as1 = defaultdict(float)
    ac = defaultdict(int)

    for i in range(i1, i2):
        g1, p1, f1 = data[i]

        for j in range(i + 1, n):
            g2, p2, f2 = data[j]
            
            # add each unique off-diagonal pair once.
            difference = abs(f1 - f2)

            dh = hdist(g1, p1, g2, p2)
            hs1[dh] += difference
            hc[dh] += 1

            da = int(np.ceil(adist_g(g1, p1, g2, p2)))
            as1[da] += difference
            ac[da] += 1

    return hs1, hc, as1, ac


def add_results(total_sum, total_count, part_sum, part_count):
    """
    Add the results of one chunk to the totals.

    input:
        total_sum, total_count - total sums and pair counts;
        part_sum, part_count - sums and pair counts from one chunk.

    output:
        None; the total dictionaries are updated in place.
    """
    for distance in part_count:
        total_sum[distance] = total_sum.get(distance, 0.0) + part_sum[distance]
        total_count[distance] = total_count.get(distance, 0) + part_count[distance]


def calculate_global_difference(data, tasks, threads=1):
    """
    Compute D(d) using the same steps as corr() in autocorrelation.

    Main idea:
        1. Compute the fitness standard deviation.
        2. Add each diagonal self-pair once, with zero difference.
        3. Process chunks of outer-loop indices i.
        4. Calculate and add every pair j > i once.
        5. Add all partial sums and pair counts together.
        6. Divide mean differences by the fitness standard deviation.

    input:
        data - loaded landscape;
        tasks - chunks created by make_tasks;
        threads - number of worker processes.

    output:
        h_df, a_df - Hamming and Atchley tables;
        columns: distance, pair_count, mean_abs_difference, D.
    """

    fitness = np.array([row[2] for row in data], dtype=float)
    sigma = np.std(fitness, ddof=0)

    # Self-pairs: N pairs with zero fitness difference.
    n = len(data)
    hs1, hc = {0: 0.0}, {0: n}
    as1, ac = {0: 0.0}, {0: n}

    init_worker(data)

    if threads == 1:
        it = map(work, tasks)

        for x_hs1, x_hc, x_as1, x_ac in tqdm(it, total=len(tasks), desc="chunks"):
            add_results(hs1, hc, x_hs1, x_hc)
            add_results(as1, ac, x_as1, x_ac)
    else:
        with mp.Pool(threads, initializer=init_worker, initargs=(data,)) as pool:
            it = pool.imap_unordered(work, tasks, chunksize=1)

            for x_hs1, x_hc, x_as1, x_ac in tqdm(it, total=len(tasks), desc="chunks"):
                add_results(hs1, hc, x_hs1, x_hc)
                add_results(as1, ac, x_as1, x_ac)

    h_df = make_df(hs1, hc, sigma)
    a_df = make_df(as1, ac, sigma)

    return h_df, a_df


#### 5. Load data and save results ####

def run_one(input_file, output_dir, threads=1, chunk_size=16):
    """
    Load one landscape, compute D(d), and save two tables.

    input:
        input_file - path to a cleaned landscape;
        output_dir - directory for results;
        threads - number of worker processes;
        chunk_size - number of outer-loop indices i in one chunk.

    output:
        h_df, a_df; also saves:
            global_difference_hamming.csv
            global_difference_atchley.csv
    """

    data, _, _ = load_data(input_file)
    tasks = make_tasks(len(data), chunk_size)
    h_df, a_df = calculate_global_difference(data, tasks, threads)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h_df.to_csv(output_dir / "global_difference_hamming.csv", index=False)
    a_df.to_csv(output_dir / "global_difference_atchley.csv", index=False)
    return h_df, a_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    args = parser.parse_args()
    run_one(args.input_file, args.output_dir, args.threads, args.chunk_size)


if __name__ == "__main__":
    main()
