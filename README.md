# MultiElo_switchable.py

Multiplayer Elo ranking for OTUs in environmental biomes.  
Two modes are available: `classic` Elo and an absence-corrected variant corresponding to the Baas-Becking score (BB-score).

## Concept

Each sample (row) is treated as one multiplayer “game”.  
OTUs listed from left to right are ordered from best to worst in that game.  
Each iteration shuffles the order of games to reduce sequence effects.

## Input

CSV matrices like `MatrixRank_MAPbiomes_<biome>.csv`.

- Rows = samples (games).  
- Columns = ordered OTU ranks (1st, 2nd, …).  
- Empty, `None`, or `NaN` cells are ignored.  
- A `date` column is allowed and ignored.

Example:

| rank1 | rank2 | rank3 |
|-------|-------|-------|
| OTU_A | OTU_B | OTU_C |

## Observed score function

Observed scores follow the exponential decay function:

$$
S_{\text{observed},i} =
\frac{\alpha^{(N - P_i)} - 1}
{\sum_{j=1}^{N} \left(\alpha^{(N - P_j)} - 1\right)}
$$

Where:

- \(N\) = number of OTUs in the game.  
- \(P_i\) = rank position of OTU \(i\) (1 = best).  
- \(\alpha\) = base coefficient (`base_coef` in the script).  
- Scores are normalized to sum to 1, and ties share the mean score.

## Expected score

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
- \(D\) = logistic scale (default 400).

## Rating update

Ratings for present OTUs are updated as:

$$
\text{Rating}_{t+1,i} =
\text{Rating}_{t,i}
+ K\left(S_{\text{observed},i} - S_{\text{expected},i}\right)
$$

Where:

- \(K\) is the gain factor (default 10).  
- \(S_{\text{observed},i}\) and \(S_{\text{expected},i}\) come from the equations above.

## Absence correction (BB-score logic)

In `corrected` mode, absent OTUs are penalized using the last-ranked present OTU.

After updating present OTUs with the equation above, the script computes:

$$
\Delta = R'_{\text{last}} - R_{\text{last}}
$$

Where:

- \(R_{\text{last}}\) = rating of the lowest-ranked present OTU **before** the update.  
- \(R'_{\text{last}}\) = rating of the same OTU **after** the update.  
- \(\Delta\) = rating change of that OTU.

This same \(\Delta\) is then added to **every absent OTU** in that game, breaking the strict zero-sum property and implementing the BB-score absence penalty.

## Install

Requires Python 3.8+.

```bash
pip install numpy pandas
