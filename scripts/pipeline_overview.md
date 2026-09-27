# Pipeline overview (case 1, FCNN)

**Learning task:** approximate the map

$$
d : \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \to \mathbb{R}_{\ge 0},
\qquad
(M, M') \mapsto W_1(M, M'),
$$

where $M, M'$ are two CO2 maps ($M_{rc}$ is the CO2 mass, in kg, in grid cell
$(r, c)$) and $W_1(M, M')$ is the Wasserstein-1 distance between them, in kg·m,
computed externally. The network is

$$
f_\theta : \mathbb{R}^{2 \times 120 \times 840} \to \mathbb{R},
\qquad
f_\theta(M, M') \approx d(M, M'),
$$

## Where $M$ and $d(M, M')$ are stored

Each map belongs to a participant $g$ (34 of them, e.g. `calgary1`) and a year
$t$, so we write it $M^{g,t}$.

- **Maps**: $M^{g,t}$ is the column `tmCO2 [kg]` of the file
  `spe11b/<g>/spe11b_spatial_map_<t>y.csv`, for $t \in \{0, 5, \dots, 1000\}$.
- **Distances**: $d(M^{g,t}, M^{g',t'})$ is the column `distance` of
  `spe11b/assembled_sorted_full.csv`, in the row with `key1` $=$ `<g>_<t>y` and
  `key2` $=$ `<g'>_<t'>y`, for all $g, g'$ and $t, t' \in \{50, 100, \dots, 1000\}$,
  also with $t \neq t'$. 

## Typical training

- Training pairs: $(M_i, M_i')$, $i = 1, \dots, N$.
- Scalings: $\mathcal{S}_M$ for maps, $\mathcal{S}_d$ for distances. Both invertible.
- Input: $x_i = \left( \mathcal{S}_M(M_i), \mathcal{S}_M(M_i') \right)$.
- Target: $y_i = \mathcal{S}_d\left( d(M_i, M_i') \right)$.
- Prediction in kg·m: $\mathcal{S}_d^{-1}\left( f_\theta(x_i) \right)$.
- Minibatch: $\mathcal{B} \subset \{1, \dots, N\}$.

Loss, mean squared error on $\mathcal{B}$:

$$
\mathcal{L}_{\mathcal{B}}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left( f_\theta(x_i) - y_i \right)^2
$$

Update at iteration $k$, with minibatch $\mathcal{B}_k$:

$$
\theta^{k+1} = \theta^k + \Delta\theta^k,
\qquad
\Delta\theta^k = G\left(\theta^k, \nabla_\theta \mathcal{L}_{\mathcal{B}_k}(\theta^k), s^k\right)
$$

- $G$: optimizer rule.
- $s^k$: optimizer state, e.g. past gradients.



## Data Management Implementation

Assumptions:

- Python, with common libraries (NumPy, JAX, optax, pandas).
- Memory is limited: each map is stored once, avoiding the common explicit storage of domain and codomain sets. This is why a proper preprocess and data-management utilities are required. 
- A pair is an index pair, never a copy of two maps.
- `spe11b/` is the raw data, never modified.

Preprocessing, run once:

$$
\text{preprocess} : \texttt{spe11b/} \mapsto \texttt{data/}
$$

- `data/` is a second starting point: training reads only `data/`, not `spe11b/`.

Content of `data/`: one HDF5 file, `data/spe11b.h5`, all unscaled.

- $\mathcal{G}$: the 34 participants.
- $\mathcal{T} = \{50, 100, \dots, 1000\}$: the years with distances.
- $K = |\mathcal{G}| \cdot |\mathcal{T}| = 680$.
- Index $k \in \{0, \dots, K-1\}$: one map. Maps and distances share it.

| Dataset | Shape | Type | Content |
|---|---|---|---|
| `maps` | $(K, 120, 840)$ | float32 | $M_k$, in kg |
| `distances` | $(K, K)$ | float64 | $D_{kk'} = d(M_k, M_{k'})$, in kg·m |
| `participant` | $(K,)$ | string | participant $g \in \mathcal{G}$ of map $k$, e.g. `calgary1` |
| `year` | $(K,)$ | int | year $t \in \mathcal{T}$ of map $k$, e.g. 50 |

- Labels: $\mu : \{0, \dots, K-1\} \to \mathcal{G} \times \mathcal{T}$, $\mu(k) = (\texttt{participant}[k], \texttt{year}[k]) = (g, t)$.
- $\mu$ is a bijection: one map per $(g, t)$.
- Map $k$: $M_k = M^{\mu(k)}$.
- Order, year-major: $k = |\mathcal{G}| \, i_t + i_g$.
- $i_g \in \{0, \dots, 33\}$: position of $g$ in $\mathcal{G}$, sorted alphabetically.
- $i_t \in \{0, \dots, 19\}$: position of $t$ in $\mathcal{T}$, sorted ascending.
- Example: $\mu(0) = (\texttt{calgary1}, 50)$, $\mu(1) = (\texttt{cau-kiel1}, 50)$, $\mu(679) = (\texttt{ut-csee2}, 1000)$.
- Redundancy: $D = D^\top$, and labels repeat. About 2 MB in total: accepted.
- Reading map $k$ loads only map $k$.

Preprocessing of the distances:

- Input: `spe11b/assembled_sorted_full.csv`, rows (`key1`, `key2`, `distance`).
- Key of map $k$: `<g>_<t>y`, with $(g, t) = \mu(k)$.
- One row sets one entry: $D_{kk'} =$ `distance`, $k$ from `key1`, $k'$ from `key2`.
- Checks: all $K^2$ entries set, $D = D^\top$, $D_{kk} = 0$.

Pairs:

- Pair: $(k, k') \in \{0, \dots, K-1\}^2$.
- Input built per minibatch, on the GPU: $x = \left( \mathcal{S}_M(M_k), \mathcal{S}_M(M_{k'}) \right)$.
- Target: $y = \mathcal{S}_d(D_{kk'})$.


TODO (?): 
create `src/ml4gcs/data_management/` (currently `src/ml4gcs/data/`), containing
all the preprocessing files, the data scaling functions and the data loader for
training.  