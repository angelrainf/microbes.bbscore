# MultiElo_switchable.py

Multiplayer Elo ranking for OTUs.  
Two modes are available: `classic` and `corrected`.  
Corrected mode adds the last-place rating change to all absent OTUs in each game.

## Concept

Each sample (row) is treated as one multiplayer “game”.  
OTUs listed left-to-right are ranked best-to-worst in that sample.  
The script shuffles the order of games in every iteration to reduce order effects.

## Input

CSV matrices like `MatrixRank_MAPbiomes_<biome>.csv`.  
Rows = samples.  
Columns = ordered OTU ranks for each sample.  
Empty, `None`, and `NaN` cells are ignored.  
A `date` column may exist and is ignored.

Example:

| rank1 | rank2 | rank3 |
|-------|-------|-------|
| OTU_A | OTU_B | OTU_C |

## Scoring and rating update

Actual scores follow an exponential rule and sum to 1:

`S_r ∝ base_coef^(n - r) - 1`

Ties share the mean score for that rank.

Expected scores come from pairwise logistic win probabilities:

`P(i beats j) = 1 / (1 + 10^(-(R_i - R_j) / D))`

Expected scores are normalized across players.

Rating update:

`R' = R + K * (S - E)`

### Absence correction (corrected mode)

After updating present OTUs, compute the last-place rating change:

`Δ = R'last − Rlast`

Add `Δ` to all OTUs absent from that game.

## Install

Requires Python 3.8+.  
Install dependencies:

```bash
pip install numpy pandas
