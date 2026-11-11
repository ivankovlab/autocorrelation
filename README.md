# Fitness landscape analysis

These scripts calculate centered autocorrelation, global fitness differences, and fitness differences relative to the wild type (WT), using Hamming and Atchley distances.

**Important!**

Keep `simple_autocorr_functions.py` in the same folder as `run_simple_autocorr.py`, `global_difference.py`, and `wt_difference.py`. These three scripts import the shared module. Run the analysis scripts, rather than the module itself.

`simple_autocorr.py` is an optional standalone implementation of autocorrelation.

## Setup

For the correct process install the following:

```bash
python -m pip install numpy pandas peptides tqdm
```

## Input
The input must be a tab-separated text file with columns named `Genotype` and `Phenotype`:

```text
Genotype	Phenotype
WT	3.719
A110D	1.301
A110G:K158R	3.659
```

Genotypes must contain substitutions between the 20 canonical amino acids. Use `WT` for the wild type and `:` to separate substitutions. Prepare the input without insertions, deletions, or stop mutations. 

In the commands below, replace `{name}` with your landscape filename without `.txt`.

## 1. Autocorrelation

Run:

```bash
python run_simple_autocorr.py {name}.txt -o results/{name}/autocorrelation --threads 1 --chunk-size 16 --max-h 25 --max-a 60
```

`simple_autocorr_functions.py` contains the calculation functions. `run_simple_autocorr.py` imports this module, runs the analysis, and saves the results.

Five tables are saved in `results/{name}/autocorrelation` for hamming, atchley, both cumulative and summary. 

- The ordinary tables contain `distance`, `pair_count`, `C`, `sd`, `sem`, `ci_lower`, `ci_upper`, and `relative_sem`.
- Cumulative tables contain `C_cum` and the cumulative count and uncertainty columns, together with the original `C` and `pair_count`.

`--max-h` and `--max-a` set the largest distances included in the corresponding tables.

To use multiple worker processes, increase `--threads`, for example to `--threads 2`. `--chunk-size 16` assigns 16 outer-loop genotype indices to each task; each index is still compared with every subsequent genotype. Chunks distribute pair calculations, and the full landscape remains in memory.

## 1*. Standalone autocorrelation

`simple_autocorr.py` calculates the same ordinary C(d) as section 1, without chunks or parallel worker processes. It is convenient for small landscapes.

Run:

```bash
python simple_autocorr.py {name}.txt -o results/{name}/simple_autocorrelation --max-h 25 --max-a 60
```

It saves three tables: `{name}_hamming.csv`, `{name}_atchley.csv`, and `{name}_summary.csv`. It does not generate cumulative tables.

## 2. Global difference

Run:

```bash
python global_difference.py {name}.txt results/{name}/global_difference --threads 1 --chunk-size 16
```

The script calculates the mean absolute fitness difference between genotypes at each distance, normalized by the fitness standard deviation of the entire input landscape:

Two tables are saved in `results/{name}/global_difference`: `global_difference_hamming.csv` and `global_difference_atchley.csv`. The columns are `distance`, `pair_count`, `mean_abs_difference`, `D`

`--threads` and `--chunk-size` work as in section 1. These are individual distance-bin tables.

## 3. WT difference

Run:

```bash
python wt_difference.py {name}.txt results/{name}/wt
```

The script compares each unique genotype with WT once and calculates:


Two tables are saved in `results/{name}/wt`: `wt_difference_hamming.csv` and `wt_difference_atchley.csv`. The columns are `distance`, `genotype_count`, `mean_abs_difference_wt`, `D_wt`

These are individual distance-bin tables. At distance zero, WT is counted once and `D_wt = 0`.

The calculation makes one pass through the loaded landscape, so it does not need `--threads` or `--chunk-size`.
