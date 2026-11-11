"""
Centered autocorrelation profiles for protein fitness landscapes.

Input files must be tab-separated and contain two columns:
    Genotype    Phenotype

Example:
    WT          3.719
    A110D       1.301
    A110G:K158R 3.659

For every input landscape, the script writes two tables:
    <landscape>_hamming.csv
    <landscape>_atchley.csv

For every input landscape, it also writes its own summary table:
    <landscape>_summary.csv
"""

import argparse
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

def make_tasks(n, chunk_size):
    """
    Split outer-loop genotype indices into chunks.

    input:
        n - number of genotypes
        chunk_size - number of indices i in one chunk

    output:
        tasks = [(i1, i2), ...]
    """

    tasks = []

    for i1 in range(0, n, chunk_size):
        i2 = min(i1 + chunk_size, n)
        tasks.append((i1, i2))

    return tasks

def corr(data, max_h=25, max_a=60):
    """    
    Compute centered autocorrelation profiles.
    
    Main idea:
        1. Center fitness: z_i = fitness_i - mean_fitness

        2. For every pair of genotypes: product = z_i * z_j

        3. Compute distance between genotypes.

        4. Add product to the corresponding distance bin.

        5. Normalize average product by fitness variance.

    input:
        data - loaded landscape: [(g, pos, fit)]

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

    for i in tqdm(range(n), desc="pairs"):
        g1, p1, _ = data[i]

        for j in range(i + 1, n):
            g2, p2, _ = data[j]

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

    h_df = make_df(hs1, hs2, hc, var)
    a_df = make_df(as1, as2, ac, var)

    stat = {
        "n": n,
        "pairs": n * (n + 1) // 2,
        "mean_fitness": float(f.mean()),
        "fitness_variance": float(var),
        "skipped_atchley_pairs": skipped_a,
    }

    return h_df, a_df, stat



#### 4. Run ####

def run_one(file, out_dir, max_h, max_a):
    file = Path(file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = file.stem # removes the extension .txt or other
    data, wt, info = load_data(file)
    print("\n" + "=" * 70)
    print(f"Landscape: {file}")
    print(f"Rows: {info['rows']:,}")
    print(f"Unique genotypes: {info['n']:,}")
    print(f"Duplicates skipped: {info['duplicates']:,}")
    n = info["n"]
    print(f"Unique pairs including self-pairs: {n * (n + 1) // 2:,}")

    h_df, a_df, stat = corr(data, max_h=max_h, max_a=max_a)

    h_path = out_dir / f"{name}_hamming.csv"
    a_path = out_dir / f"{name}_atchley.csv"
    s_path = out_dir / f"{name}_summary.csv"

    h_df.to_csv(h_path, index=False)
    a_df.to_csv(a_path, index=False)

    summary = {
        "landscape": name,
        "input_file": str(file),
        "rows_in_file": info["rows"],
        "unique_genotypes": info["n"],
        "duplicates_skipped": info["duplicates"],
        **stat,
        "wt_positions": len(wt),
        "max_hamming": max_h,
        "max_atchley": max_a,
        "atchley_binning": "ceil",
        "hamming_table": str(h_path),
        "atchley_table": str(a_path),
    }

    pd.DataFrame([summary]).to_csv(s_path, index=False)

    print(f"Saved: {h_path}")
    print(f"Saved: {a_path}")
    print(f"Saved: {s_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("-o", "--out", default="autocorr_results")
    p.add_argument("--max-h", type=int, default=25)
    p.add_argument("--max-a", type=int, default=60)

    return p.parse_args()


def main():
    args = get_args()
    for file in args.files:
        run_one(file=file, out_dir=args.out, max_h=args.max_h, max_a=args.max_a)


if __name__ == "__main__":
    main()
