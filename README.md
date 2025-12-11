# MultiElo_switchable.py

Multiplayer Elo ranking for OTUs.  
Two modes are available: `classic` and `corrected`.  
Corrected mode introduces an absence penalty identical to the Baas-Becking score (BB-score) concept.

## Concept

Each sample (row) is treated as one multiplayer “game”.  
OTUs listed left-to-right are ranked best-to-worst in that sample.  
Each iteration shuffles game order to reduce sequence effects.

## Input

CSV matrices like `MatrixRank_MAPbiomes_<biome>.csv`.  
Rows = samples.  
Columns = ordered OTU ranks.  
Empty, `None`, or `NaN` cells are ignored.  
A `date` column is ignored.

Example:

| rank1 | rank2 | rank3 |
|-------|-------|-------|
| OTU_A | OTU_B | OTU_C |

## Scoring function

Observed scores follow the exponential decay function used in the manuscript:
$$
\[
S_{\text{observed},i}
= \frac{\alpha^{(N-P_i)} - 1}{\sum_{j=1}^{N} \left( \alpha^{(N-P_j)} - 1 \right)}
\tag{1}
\]
$$
Where:  
- \(N\) = number of present OTUs in the game,  
- \(P_i\) = rank position of OTU \(i\),  
- \(\alpha\) = base coefficient,  
- Scores sum to 1 and ties share the mean score.

## Expected scores

Expected scores follow the logistic formulation:

\[
S_{\text{expected},i}
= \frac{\sum_{j \neq i} \frac{1}{1 + 10^{\frac{(R_j - R_i)}{D}}}}
{n (n - 1)/2}
\tag{3}
\]

Where:  
- \(n\) = number of OTUs in the match,  
- \(R_i\) = rating of OTU \(i\),  
- \(D\) = logistic scale (default 400).

## Rating update

Ratings for present OTUs are updated as:

\[
\text{Rating}_{t+1,i}
= \text{Rating}_{t,i}
+ K \left( S_{\text{observed},i} - S_{\text{expected},i} \right)
\tag{2}
\]

Where \(K\) is the gain factor (default 10).

## Absence correction (BB-score logic)

The script implements the absence penalty exactly as used in the BB-score method.

After updating present OTUs using equation (2), compute the rating change of the last-ranked present OTU:

\[
\Delta = R'_{\text{last}} - R_{\text{last}}
\]

Where:  
- \(R_{\text{last}}\) = rating of the lowest-ranked present OTU before updating,  
- \(R'_{\text{last}}\) = rating of the same OTU after updating,  
- \(\Delta\) = rating change applied to **all absent OTUs**.

Every absent OTU receives this same \(\Delta\), breaking the zero-sum constraint and producing BB-scores.

## Install

Requires Python 3.8+.  
Dependencies:

```bash
pip install numpy pandas
