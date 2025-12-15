# Repository to reproduce results from manuscript Rain-Franco, A., Andrei AS & Pernthaler, J. 2026.
Rmarkdown files and supporting material to reproduce results and statistical analyses of publication. 

Repository includes:
-Elo-rating input
-Occurrence data
-Tutorial for Elo-rating and Bass-Becking score (BB-score) calculation.

## MultiElo_switchable.py

Multiplayer Elo ranking for OTUs in environmental biomes.
Two modes are available: classic Elo and an absence-corrected variant corresponding to the Baas-Becking score (BB-score).

### Concept

Each sample (row) is treated as one multiplayer “game”.
OTUs listed from left to right are ordered from best to worst in that game.
Each iteration shuffles the order of games to reduce sequence effects.

### Input

CSV matrices like MatrixRank_MAPbiomes_<biome>.csv.
- Rows = samples (games)  
- Columns = ordered OTU ranks (1st, 2nd, …)  
- Empty, `None`, or `NaN` cells are ignored  
- A `date` column is allowed and ignored

Example:

|   rank1  |  rank2   |   rank3  |   rank4  |   rank5  |
|----------|----------|----------|----------|----------|
| Specie_A | Specie_B | Specie_C | Specie_D | Specie_E |

### Observed score function

Observed scores follow an exponential decay function:


$$
S_{\text{observed},i} =
\frac{\alpha^{(N - P_i)} - 1}
{\sum_{j=1}^{N} \left(\alpha^{(N - P_j)} - 1\right)}
$$

Where:
- '\N\' = number of OTUs in the game.  
- '\P\_i\' = rank position of OTU \(i\) (1 = best).  
- '\alpha\' = base coefficient .  
- Scores are normalized to sum to 1, and ties share the mean score.

### Expected score

Expected scores use pairwise logistic win probabilities:

$$
S_{\text{expected},i} =
\frac{\sum_{j \ne i} \frac{1}{1 + 10^{(R_j - R_i)/D}}}
{n (n - 1)/2}
$$

Where:
- \(n\) = number of OTUs in the game.  
- \(R_i\) = current rating of OTU \(i\).  
- \(R_j\) = current rating of competitor \(j\).  
- \(D\) = logistic scale (default 400)

### Rating update

$$
\text{Rating}_{t+1,i} =\text{Rating}_{t,i}+ K\left(S_{\text{observed},i} - S_{\text{expected},i}\right)
$$

Where:
- K = gain factor (default 10)
- $$\(S_{\text{observed},i}\) and \(S_{\text{expected},i}\)$$ come from the equations above.

### Absence correction (BB-score logic)

In corrected mode, absent OTUs are penalized using the last-ranked present OTU.

$$
\Delta = R'_{\text{last}} - R_{\text{last}}
$$

$$R_{\text{last}}$$ = rating of the lowest-ranked present OTU before updating\
$$R'_{\text{last}}$$  = rating after updating\
$$\Delta$$ is added to all absent OTUs.\

This breaks zero-sum Elo and implements BB-score absence penalization.

## Install

Requires Python 3.8+.

pip install numpy pandas

### Usage
Run on a directory:
```bash
python3 MultiElo_switchable.py \
  --mode classic \
  --input MatrixForElo \
  --output results
```




### Main command-line options

`--mode classic|corrected`
`--input <directory or glob>` \
`--output <directory>` \
`--subsample <int>` \
`--iters <int>` \
`--workers <int> `\
`--seed <int> `\
`--K <float>` \
`--D <float> `\
`--coef-file <csv> `\
`--coef-value <float>` \

### Output
One CSV per biome:
`rank, player_id, n_games, rating, run, biome`
