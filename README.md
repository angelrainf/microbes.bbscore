
## Repository to reproduce results from manuscript "Macroecology of microbial performance across Earth’s biomes" by Rain-Franco, A., Andrei A.S., & Pernthaler, J. 2026.

Rmarkdown files and supporting material to reproduce results and statistical analyses of publication. 

Repository includes:

-Elo-rating data (average+sd) after interations.\
-Occurrence data\
-Tutorial for Elo-rating and Bass-Becking score (BB-score) calculation.

## MultiElo_switchable.py

Multiplayer Elo ranking for OTUs in environmental biomes.
Two modes are available: classic Elo and an absence-corrected variant corresponding to the Baas-Becking score (BB-score).

### Concept

Each sample (row) is treated as one multiplayer “game”.
OTUs listed from left to right are ordered from best to worst in that game.
Each iteration shuffles the order of games to reduce sequence effects.

### Input
CSV matrices like MatrixRank_<group of samples>.csv.
Each row of the input matrix is treated as a multiplayer game based on relative abundances: OTUs are ordered from left (highest relative abundance) to right (lowest), and empty, None, or NaN cells are ignored.

- Rows = samples (games)  
- Columns = ordered OTU ranks (1st, 2nd, …)  
- Empty, `None`, or `NaN` cells are ignored  
- A `date` column is allowed and ignored

Example:

|   rank1  |  rank2   |   rank3  |   rank4  |   rank5  |
|----------|----------|----------|----------|----------|
| Specie_A | Specie_B | Specie_C | Specie_D | Specie_E |

### Observed score function
For calcuation, a scoring function is defined by a fitted decay function to the distribution of the species observed (within a group of samples) follow an exponential decay function:


$$
S_{\text{observed},i} =
\frac{\alpha^{(N - P_i)} - 1}
{\sum_{j=1}^{N} \left(\alpha^{(N - P_j)} - 1\right)}
$$

Where:
- $$N$$= number of OTUs in the game.  
- $$P\_i$$ = rank position of OTU \(i\) (1 = best).  
- $$alpha$$ = base coefficient .  
- Scores are normalized to sum to 1, and ties share the mean score.

### Expected score

The expected score is calculated given the current ratings of all participants divided by the total number of pairwise interactions:

$$
S_{\text{expected},i} =
\frac{\sum_{j \ne i} \frac{1}{1 + 10^{(R_j - R_i)/D}}}
{n (n - 1)/2}
$$

Where:
- $$n$$ = number of OTUs in the game.  
- $$R_i$$ = current rating of OTU \(i\).  
- $$R_j$$ = current rating of competitor \(j\).  
- $$D$$ = logistic scale (default 400)

### Rating update
After each match, the score of each species is updated following:

$$
\text{Rating}_{t+1,i} =\text{Rating}_{t,i}+ K\left(S_{\text{observed},i} - S_{\text{expected},i}\right)
$$

Where:
- K = gain factor (default 10)
- $$\(S_{\text{observed},i}\)$$ and $$\(S_{\text{expected},i}\)$$ come from the equations above.

### Absence correction (BB-score logic)
Our proposed variation of rating score considers the relative abundance (performance) of all species present in a given sample for updating their rating, but introduces a penalization of species detected in the biome pool (or “tournament”) but absent from the focal community (“match”).\
So, each absent species will be assigned the rank of the lowest-performing species in the focal sample and its overall rank was updated accordingly. This absence-correction breaks the  zero-sum property of Elo-rating.\
Because this calculation assumes that every species occurring in a biome could potentially be present in every sample, but, “the environment selects” (reflecting the famous postulate by L. Baas-Becking), we named this absence-corrected variant of the Elo-rating the “Baas-Becking-score” (BB-score). 

The absent species are penalized using the last-ranked present species.

$$
\Delta = R'_{\text{last}} - R_{\text{last}}
$$

$$R_{\text{last}}$$ = rating of the lowest-ranked present OTU before updating\
$$R'_{\text{last}}$$  = rating after updating\
$$\Delta$$ is added to all absent OTUs.

## Install

Requires Python 3.8+.

pip install numpy pandas

### Simple Usage
Run on a directory:
```bash
python3 MultiElo_switchable.py \
  --mode classic \
  --input MatrixForElo \
  --output results
```


### Main command-line options

`--mode classic|corrected`
`--input <directory>` \
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
`rank, player_id, n_games, rating_calculated, run_iteration, file_name`
