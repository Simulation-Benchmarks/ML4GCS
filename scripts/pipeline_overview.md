# Pipeline overview

**Learning task:** approximate the map

$$
d : \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \to \mathbb{R}_{\ge 0},
\qquad
(M, M') \mapsto W(M, M'),
$$

where $M, M'$ are two CO2 maps, in kg, and $W(M, M')$ is the Wasserstein
distance between them, in kg·m, computed externally. The network is

$$
f_\theta : \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \to \mathbb{R},
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
- Key notation: `<·>` is a placeholder; `_` and `y` (years) are literal, e.g. `calgary1_100y`.

## Typical training

- Training pairs: $(M_i, M_i')$, $i = 0, \dots, N-1$, where $N$ is the size of the training dataset.
- Scalings: $\mathcal{S}_M$ for maps, $\mathcal{S}_d$ for distances. Both invertible.
- Input: $x_i = \left( \mathcal{S}_M(M_i), \mathcal{S}_M(M_i') \right)$.
- Target: $y_i = \mathcal{S}_d\left( d(M_i, M_i') \right)$.
- Prediction in kg·m: $\mathcal{S}_d^{-1}\left( f_\theta(x_i) \right)$.
- Minibatch: $\mathcal{B} \subset \{0, \dots, N-1\}$.

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
- We want to keep the string-names of the partecipant instead of replaccing them with indeces. Similarly for timesteps. 


**Indexing:**

- $\mathcal{G}= \{g_0, \dots, g_{|\mathcal{G}|-1}\} = \{\text{calgary1},\ldots, \text{ut-csee2} \}$: the names of the 34 participants.
- $i_g \in \{0, \dots, |\mathcal{G}|-1\}$: numerical index
- $\mathcal{T} = \{t_0, \dots, t_{|\mathcal{T}|-1}\} = \{50, 100, \dots, 1000\}$: the years with distances.
- $i_t \in \{0, \dots, |\mathcal{T}|-1\}$: numerical index
- $K = |\mathcal{G}| \cdot |\mathcal{T}| = 680$.

**From double-index to unrolled index (maybe unnecessary):**

- $k \in \{0, \dots, K-1\}$: unrolled index, participant index moves faster:

$$
\mathcal{K} : \{0, \dots, |\mathcal{G}|-1\} \times \{0, \dots, |\mathcal{T}|-1\} \to \{0, \dots, K-1\}, \qquad \mathcal{K}(i_g, i_t) = |\mathcal{G}| \, i_t + i_g
$$


**From linear index to coupled unrolled index:**


$$
\pi : \{0, \dots, N-1\} \to \{0, \dots, K-1\}^2,
\qquad
\pi(i) = (k_i, k_i')
$$



**From unrolled index to label:**

- $\lambda = (g, t) \in \mathcal{G} \times \mathcal{T}$: label of a map.

$$
\Lambda : \{0, \dots, K-1\} \to \mathcal{G} \times \mathcal{T},
\qquad
\Lambda(k) = \left( g_{i_g}, t_{i_t} \right),
\quad
i_g = k \bmod |\mathcal{G}|,
\quad
i_t = \lfloor k / |\mathcal{G}| \rfloor
$$

- E.g., $\Lambda(k) = (\texttt{participant}[k], \texttt{year}[k])$; $\Lambda(0) = (\texttt{calgary1}, 50)$, $\Lambda(1) = (\texttt{cau-kiel1}, 50)$. (Here is the assumption of using string-names instead of indices.)




**Preprocessing, run once:**

$$
\text{preprocess} : \texttt{spe11b/} \mapsto \texttt{HDF5 file}
$$

- HDF5 file store in `data/`. Is a second starting point: training reads only `data/`, not `spe11b/`.

Content of `data/`: one HDF5 file, `data/spe11b.h5`, all unscaled. `data/spe11b.h5` is everything the user needs, alongside the provided utilities to select specific data from it for training. Why HDF5 format? Because it is common and relatively easy to use. 


Content of HDF5 file:
| Dataset | Shape | Type | Content |
|---|---|---|---|
| `maps` | $(K, 120, 840)$ | float32 | $M_k$, in kg |
| `distances` | $(K, K)$ | float64 | $D_{kk'} = d(M_k, M_{k'})$, in kg·m |
| `participant` | $(K,)$ | string | participant $g \in \mathcal{G}$ of map $k$, e.g. `calgary1` |
| `year` | $(K,)$ | int | year $t \in \mathcal{T}$ of map $k$, e.g. 50 |



Redundancy:
 
- $D = D^\top$, and labels repeat. Estimated 2 MB in total: accepted.



**Data management utilities:**

- ML needs fast minibatch creation.
- The HDF5 file is kept in memory.
- Data is picked on the fly, when a minibatch is built.
- Required: a function, $h$, from an linear index to a training sample:

$$
h : \{0, \dots, N-1\} \to \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \times \mathbb{R}_{\ge 0},
\qquad
h : i \mapsto \left( (M_{k_i}, M_{k'_i}), D_{k_i k'_i} \right)
$$


- $\mu$: from a label pair to a training sample, with $M^{\lambda} = M^{g,t}$ for $\lambda = (g, t)$:

$$
\mu : (\mathcal{G} \times \mathcal{T})^2 \to \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \times \mathbb{R}_{\ge 0},
\qquad
\mu : (\lambda, \lambda') \mapsto \left( (M^{\lambda}, M^{\lambda'}), d(M^{\lambda}, M^{\lambda'}) \right)
$$

- We have:

$$
h = \mu \circ (\Lambda \times \Lambda)  \circ \pi,
$$


- Minibatch $\mathcal{B}$: $\left\{ h(i) : i \in \mathcal{B} \right\}$.
