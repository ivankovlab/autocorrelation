"""
Functions for centered autocorrelation profiles of protein fitness landscapes.

Input files must be tab-separated and contain two columns:
    Genotype    Phenotype

Example:
    WT          3.719
    A110D       1.301
    A110G:K158R 3.659

The functions are called from run_simple_autocorr.py.
"""

import multiprocessing as mp
import re
from pathlib import Path
import numpy as np
import pandas as pd
import peptides
from tqdm.auto import tqdm


#### 1. Load and parse landscapes ####

def parse_g(g):
    """
    Convert one genotype string into a dictionary.

    input:
        g = "A101D:C302E"

    output:
        muts = {101: ("A", "D"), 302: ("C", "E")}
        "WT" -> {}

    The function also ignores synonymous mutations:
        "A101A" -> {}
        "A101A:C102D" -> {102: ("C", "D")}
    """

    g = str(g).strip().upper()

    if g == "WT":
        return {}

    if not g:
        raise ValueError("Empty genotype. Use WT for the wild type.")

    muts = {}

    for x in g.split(":"):
        x = x.strip()

        wt = x[0]
        pos = int(x[1:-1])
        aa = x[-1]

        # If preprocessing was not done, this prevents A101A from counting.
        if wt == aa:
            continue

        muts[pos] = (wt, aa)

    return muts



def load_data(file):
    """
    Load one tab-separated landscape.

    input:
        file - path to a landscape file.

        The file must contain two columns:
            Genotype    Phenotype

        Example:
            WT              3.719
            A110D           1.301
            A110G:K158R     3.659

    output:
        data = [(genotype_dict, position_set, fitness), ...]
        wt   = {position: wild_type_aa}
        info = short summary
    """

    file = Path(file)

    df = pd.read_csv(file, sep="\t")
    df = df[["Genotype", "Phenotype"]]

    data = []
    seen = set()
    wt = {}
    dup = 0 # counting duplicated genotypes

    for _, row in df.iterrows():
        g = parse_g(row["Genotype"])
        key = tuple(sorted(g.items()))

        if key in seen:
            dup += 1
            continue

        seen.add(key)

        for p, (wt_aa, _) in g.items():
            if p not in wt:
                wt[p] = wt_aa

        pos = set(g.keys())
        fit = float(row["Phenotype"])

        data.append((g, pos, fit))

    info = {
        "rows": len(df),
        "n": len(data),
        "duplicates": dup,
    }

    return data, wt, info


#### 2. Distances ####

#### Hamming ####

def hdist(g1, p1, g2, p2, max_d=None):
    """
    Compute Hamming distance between two genotypes.

    input:
        g1, g2 - genotype dictionaries:{position: (wt_aa, new_aa)}
        p1, p2 - sets of mutated positions: {position1, position2, ...}
        max_d - optional maximum distance

    output:
        d - Hamming distance

    Logic:
        1. If a position is mutated only in one genotype, it adds 1.
        2. If a position is mutated in both genotypes but amino acids differ, it adds 1.
        3. If both genotypes have the same amino acid at the same position, it adds 0.
    """

    d = len(p1 ^ p2)

    if max_d is not None and d > max_d:
        return max_d + 1

    for p in p1 & p2:
        if g1[p][1] != g2[p][1]:
            d += 1

            if max_d is not None and d > max_d:
                return max_d + 1

    return d

#### Atchley ####

AA = set("ACDEFGHIKLMNPQRSTVWY")
AF = {aa: np.asarray(peptides.Peptide(aa).atchley_factors(), dtype=float) for aa in AA}
AD = {(a, b): float(np.linalg.norm(AF[a] - AF[b])) for a in AA for b in AA}

def adist_aa(a, b):
    """
    Compute Atchley distance between two amino acids.

    input:
        a, b - amino acid letters (str, "A")
    output:
        d - Euclidean distance between Atchley factor vectors.

    notes:
        If amino acids are identical, distance is 0.
        If one of the amino acids is non-standard, distance is NaN.
    """

    a = a.upper()
    b = b.upper()

    if a == b:
        return 0.0

    if a not in AA or b not in AA:
        return np.nan

    return AD[(a, b)]


def adist_g(g1, p1, g2, p2, max_d=None):
    """
    Compute Atchley distance between two genotypes.

    input:
        g1, g2 - genotype dictionaries: {position: (wt_aa, new_aa)}
        p1, p2 - sets of mutated positions.
        max_d - optional maximum distance.

    output:
        d - total Atchley distance between two genotypes.

    Logic:
        For every position where genotypes differ:

        1. If both genotypes are mutated at this position - compare new amino acid from g1 with new amino acid from g2.

        2. If only g1 is mutated - compare new amino acid from g1 with WT amino acid.

        3. If only g2 is mutated - compare WT amino acid with new amino acid from g2.
    """

    d = 0.0

    for p in p1 | p2:
        if p in g1 and p in g2:
            a = g1[p][1]
            b = g2[p][1]

        elif p in g1:
            a = g1[p][1]
            b = g1[p][0]

        else:
            a = g2[p][0]
            b = g2[p][1]

        x = adist_aa(a, b)
        d += x

        if max_d is not None and d > max_d:
            return max_d + 1

    return d


#### 3. Correlation table ####

def make_df(s1, s2, cnt, var):
    
    """
    Convert  pair statistics into a result table.
    
    input:
        s1 - array s1[d] = sum(z_i * z_j) for distance d

        s2 - array s2[d] = sum((z_i * z_j)^2) for distance d

        cnt - cnt[d] = number of genotype pairs at distance d

        var - fitness variance: var = mean((fitness - mean_fitness)^2)

    output:
        df - pandas DataFrame with columns:
            {distance, pair_count, C, sd, sem, ci_lower, ci_upper, relative_sem}
    """

    D = len(cnt)

    C = np.full(D, np.nan)
    sd = np.full(D, np.nan)
    sem = np.full(D, np.nan)

    for d in range(D):
        if cnt[d] == 0:
            continue

        m1 = s1[d] / cnt[d]
        m2 = s2[d] / cnt[d]

        C[d] = m1 / var

        v = max(m2 - m1**2, 0.0)
        sd[d] = np.sqrt(v) / var
        sem[d] = sd[d] / np.sqrt(cnt[d])

    out = pd.DataFrame({
        "distance": np.arange(D),
        "pair_count": cnt,
        "C": C,
        "sd": sd,
        "sem": sem,
    })

    out["ci_lower"] = out["C"] - 1.96 * out["sem"]
    out["ci_upper"] = out["C"] + 1.96 * out["sem"]
    out["relative_sem"] = out["sem"] / out["C"].abs()

    return out


#### 4. Multiprocessing ####

def make_tasks(n, chunk_size):
    """
    Split outer-loop genotype indices into chunks.

    input:
        n - number of genotypes
        chunk_size - number of outer-loop indices i in one chunk

    output:
        tasks = [(i1, i2), ...]

        i1 - first index included in a chunk
        i2 - first index not included in a chunk

    Example:
        n = 10 and chunk_size = 4 produce:
            [(0, 4), (4, 8), (8, 10)]
            
    IMPORTANT: Splitting indices into chunks does not split the landscape into
    independent parts. For every i, all pairs (i, j), where j > i,
    are still calculated.
    """

    tasks = []

    for i1 in range(0, n, chunk_size):
        i2 = min(i1 + chunk_size, n)
        tasks.append((i1, i2))

    return tasks


_DATA = None
_Z = None
_MAX_H = None
_MAX_A = None


def init_worker(data, z, max_h, max_a):
    """
    Save common data inside each worker process.

    input:
        data - loaded landscape: [(g, pos, fit)]
        z - array of centered fitness values
        max_h - maximum Hamming distance
        max_a - maximum Atchley distance

    output:
        None

    The function is called once when a worker process starts.
    It saves the common data in global variables, so the complete
    landscape does not have to be sent again with every task.

    Each process has its own global variables. The processes only read
    these variables and do not change the landscape or fitness values.
    """

    global _DATA, _Z, _MAX_H, _MAX_A

    _DATA = data
    _Z = z
    _MAX_H = max_h
    _MAX_A = max_a


def work(task):
    """
    Calculate pair statistics for one chunk of genotype indices.

    input:
        task = (i1, i2)

        i1 - first outer-loop index included in this task
        i2 - first outer-loop index not included in this task

        For every i in range(i1, i2), the function compares genotype i
        with all genotypes j in range(i + 1, n).

    output:
        hs1 - Hamming sums of z_i * z_j for this chunk
        hs2 - Hamming sums of (z_i * z_j)^2 for this chunk
        hc - Hamming pair counts for this chunk
        as1 - Atchley sums of z_i * z_j for this chunk
        as2 - Atchley sums of (z_i * z_j)^2 for this chunk
        ac - Atchley pair counts for this chunk
        skipped_a - number of Atchley pairs skipped in this chunk

    Every process creates its own arrays. Therefore, different processes
    never write into the same arrays at the same time. The main process
    later adds all partial arrays together.
    """

    i1, i2 = task

    data = _DATA
    z = _Z
    max_h = _MAX_H
    max_a = _MAX_A

    n = len(data)
    H = max_h + 1
    A = max_a + 1

    hs1 = np.zeros(H)
    hs2 = np.zeros(H)
    hc = np.zeros(H, dtype=np.int64)

    as1 = np.zeros(A)
    as2 = np.zeros(A)
    ac = np.zeros(A, dtype=np.int64)

    skipped_a = 0

    for i in range(i1, i2):
        g1, p1, _ = data[i]

        for j in range(i + 1, n):
            g2, p2, _ = data[j]

            # Add each unique off-diagonal pair once.
            prod = z[i] * z[j]
            prod2 = prod * prod

            dh = hdist(g1, p1, g2, p2, max_d=max_h)

            if dh <= max_h:
                hs1[dh] += prod
                hs2[dh] += prod2
                hc[dh] += 1

            da = adist_g(g1, p1, g2, p2, max_d=max_a)

            if not np.isfinite(da):
                skipped_a += 1
                continue

            if da <= max_a:
                da = int(np.ceil(da))
                as1[da] += prod
                as2[da] += prod2
                ac[da] += 1

    return hs1, hs2, hc, as1, as2, ac, skipped_a


def corr(data, tasks, max_h=25, max_a=60, threads=1):
    """    
    Compute centered autocorrelation profiles.
    
    Main idea:
        1. Center fitness: z_i = fitness_i - mean_fitness

        2. Add every diagonal self-pair once.

        3. Calculate and add every unique off-diagonal pair once (i < j).

        4. Compute distance between genotypes.

        5. Add the product to the corresponding distance bin.

        6. Normalize average product by fitness variance.

    input:
        data - loaded landscape: [(g, pos, fit)]
        max_h - maximum Hamming distance
        max_a - maximum Atchley distance
        threads - number of parallel worker processes
        tasks - chunks created before calling corr()

    Parallel calculation:
        1. Split the outer-loop indices i into chunks.

        2. Give the chunks to worker processes.

        3. Each worker calculates its own partial sums and pair counts.

        4. Add all partial results to the main arrays.

        All pairs are still included exactly once. Splitting the indices
        into chunks does not split the landscape into independent parts.

        The resulting sums and counts represent the upper triangle of the
        pair matrix, including the diagonal: n * (n + 1) / 2 pairs.

    output:
        h_df - Hamming autocorrelation table.
        a_df - Atchley autocorrelation table.
        stat - dictionary with calculation statistics:
            {
                "n": number of genotypes,
                "pairs": number of genotype pairs,
                "mean_fitness": mean fitness,
                "fitness_variance": fitness variance,
                "skipped_atchley_pairs": number of pairs skipped for Atchley,
            }
    """

    n = len(data)

    f = np.asarray([x[2] for x in data], dtype=float)
    z = f - f.mean()
    var = np.mean(z**2)

    H = max_h + 1
    A = max_a + 1

    hs1 = np.zeros(H)
    hs2 = np.zeros(H)
    hc = np.zeros(H, dtype=np.int64)

    as1 = np.zeros(A)
    as2 = np.zeros(A)
    ac = np.zeros(A, dtype=np.int64)

    # distance 0: genotype with itself
    self_prod = z**2

    hs1[0] = self_prod.sum()
    hs2[0] = (self_prod**2).sum()
    hc[0] = n

    as1[0] = self_prod.sum()
    as2[0] = (self_prod**2).sum()
    ac[0] = n

    skipped_a = 0

    init_worker(data, z, max_h, max_a)

    if threads <= 1:
        it = map(work, tasks)

        for res in tqdm(it, total=len(tasks), desc="chunks"):
            x_hs1, x_hs2, x_hc, x_as1, x_as2, x_ac, x_skip = res

            hs1 += x_hs1
            hs2 += x_hs2
            hc += x_hc
            as1 += x_as1
            as2 += x_as2
            ac += x_ac
            skipped_a += x_skip

    else:
        with mp.Pool(
            processes=threads,
            initializer=init_worker,
            initargs=(data, z, max_h, max_a),
        ) as pool:
            it = pool.imap_unordered(work, tasks, chunksize=1)

            for res in tqdm(it, total=len(tasks), desc=f"chunks ({threads} processes)"):
                x_hs1, x_hs2, x_hc, x_as1, x_as2, x_ac, x_skip = res

                hs1 += x_hs1
                hs2 += x_hs2
                hc += x_hc
                as1 += x_as1
                as2 += x_as2
                ac += x_ac
                skipped_a += x_skip

    h_df = make_df(hs1, hs2, hc, var)
    a_df = make_df(as1, as2, ac, var)

    stat = {
        "n": n,
        "pairs": n * (n + 1) // 2,
        "mean_fitness": float(f.mean()),
        "fitness_variance": float(var),
        "skipped_atchley_pairs": skipped_a,
        "matrix_part": "upper triangle including diagonal; each pair once",
    }

    return h_df, a_df, stat


#### 5. Cumulative autocorrelation ####

def make_cumulative_autocorr(df):
    """
    Calculate cumulative autocorrelation using unique pairs with distance <= D.

    Self-pairs are included once. Over the complete landscape, the final
    value is 1 / (n + 1), rather than zero, under this counting convention.

    input:
        df - autocorrelation table with columns:
            distance, pair_count, C, sd

    output:
        out - cumulative autocorrelation table
    """

    df = df[["distance", "pair_count", "C", "sd"]].copy()

    df = df[df["pair_count"] > 0]
    df = df.dropna(subset=["C", "sd"])
    df = df.sort_values("distance")

    cnt = df["pair_count"].to_numpy()
    C = df["C"].to_numpy()
    sd = df["sd"].to_numpy()

    cum_cnt = np.cumsum(cnt)
    cum_C = np.cumsum(cnt * C) / cum_cnt

    mean_x2 = sd**2 + C**2
    cum_mean_x2 = np.cumsum(cnt * mean_x2) / cum_cnt

    cum_var = cum_mean_x2 - cum_C**2
    cum_var = np.maximum(cum_var, 0)

    cum_sd = np.sqrt(cum_var)
    cum_sem = cum_sd / np.sqrt(cum_cnt)

    out = pd.DataFrame({
        "distance": df["distance"],
        "pair_count": cnt,
        "pair_count_cum": cum_cnt,
        "C": C,
        "C_cum": cum_C,
        "sd_cum": cum_sd,
        "sem_cum": cum_sem,
        "ci_lower": cum_C - 1.96 * cum_sem,
        "ci_upper": cum_C + 1.96 * cum_sem,
    })

    return out
