## Assumptions

- Python with common libraries (NumPy, JAX, optax, pandas).
- Memory is limited: each map is stored once, instead of the common practice of storing every input pair explicitly. This requires a preprocessing step and data-management utilities.
- A pair is an index pair, never a copy of two maps.
- `spe11b/` is the raw data, never modified.
- Participants keep their names (e.g. `calgary1`) instead of being replaced by indices. The same holds for years (e.g. 50).


## Learning task

Learning task: approximate the function

$$
d : \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \to \mathbb{R}_{\ge 0},
\qquad
(M, \tilde M) \mapsto W_{\text{as}}(M, \tilde M),
$$

where $M, \tilde M$ are two CO2 maps, in kg, and $W_{\text{as}}(M, \tilde M)$ is the Wasserstein
distance between them, in kg·m, computed externally. The network is

$$
f_\theta : \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \to \mathbb{R},
\qquad
f_\theta\left( \mathcal{S}_M(M), \mathcal{S}_M(\tilde M) \right) \approx \mathcal{S}_d\left( d(M, \tilde M) \right),
$$

with the scalings $\mathcal{S}_M$, $\mathcal{S}_d$ defined below.


Each map depends on a participant $p$ and a year $t$. For clarity, we denote it $M_{p,t}$.


## Datasets

Let $\mathcal{D}_{\text{train}}$ be the training dataset, $\mathcal{D}_{\text{test}}$ the test dataset and $\mathcal{D}_{\text{val}}$ the validation dataset.

- Each dataset is a set of samples $\left( (M, \tilde M), d(M, \tilde M) \right)$: two maps and the distance between them.
- $N = |\mathcal{D}_{\text{train}}|$: number of training samples.
- $i \in \{0, \dots, N-1\}$: training index, identifies one sample of $\mathcal{D}_{\text{train}}$.



## Scaling
- Scalings: $\mathcal{S}_M$ for maps, $\mathcal{S}_d$ for distances. Both invertible.

Example, standardization, as in `scripts/case_3_cnn_oop/data/dataset.py`:

$$
\mathcal{S}_M(M) = \frac{M - \mu_M}{\sigma_M + \varepsilon},
\qquad
\mathcal{S}_d\left( d(M, \tilde M) \right) = \frac{d(M, \tilde M) - \mu_d}{\sigma_d + \varepsilon}
$$

- $\mu_M, \sigma_M$: mean and standard deviation of all entries of all maps, not only those in $\mathcal{D}_{\text{train}}$. Two scalars, the same for every entry of $M$.
- $\mu_d, \sigma_d$: mean and standard deviation of the distances in $\mathcal{D}_{\text{train}}$.
- $\varepsilon = 10^{-8}$: avoids division by zero.
- $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test}}$: same $\mathcal{S}_M$, $\mathcal{S}_d$ as $\mathcal{D}_{\text{train}}$.
- Inverse: $\mathcal{S}_d^{-1}(y) = (\sigma_d + \varepsilon) \, y + \mu_d$.





## Minibatching

Let $g(i)$ ($g$ as in "getitem") be the scaled training sample identified by the training index $i$.

- Minibatch dataset: $\left\{ g(i) : i \in \mathcal{B} \right\}$.
- Minibatch index set: $\mathcal{B} \subseteq \{0, \dots, N-1\}$.
- $n = 0, 1, \ldots$: training iteration index (one iteration corresponds to one weight update).
- $B$: batch size.
- $J = \lceil N / B \rceil$: number of minibatches per epoch. The last one has $N - (J-1) \, B \le B$ samples.
- $I = (I_0, \dots, I_{N-1})$: training index sequence, initially $(0, \dots, N-1)$.
- $\text{reshuffle}(I)$: the entries of $I$ in a uniformly random order.

Each epoch:

1. If the user enables reshuffle: $I \leftarrow \text{reshuffle}(I)$.
2. For $j = 0, \dots, J-1$: run iteration $n$ with $\mathcal{B}^{(n)}$ below, then $n \leftarrow n + 1$.

$$
\mathcal{B}^{(n)} = \left\{ I_m : j \, B \le m < \min\left( (j+1) \, B, N \right) \right\}
$$

Each epoch uses every training index exactly once.



## Typical training

- $\lambda_i = (p_i, t_i)$, $\tilde\lambda_i = (\tilde p_i, \tilde t_i)$: participant and year of the two maps of training sample $i$.
- Training pairs: $(M_{\lambda_i}, M_{\tilde\lambda_i})$, $i = 0, \dots, N-1$.
- Input: $x_i = \left( \mathcal{S}_M(M_{\lambda_i}), \mathcal{S}_M(M_{\tilde\lambda_i}) \right)$.
- Target: $y_i = \mathcal{S}_d\left( d(M_{\lambda_i}, M_{\tilde\lambda_i}) \right)$.
- Prediction in kg·m: $\mathcal{S}_d^{-1}\left( f_\theta(x_i) \right)$.


Loss, mean squared error on a minibatch:

$$
\mathcal{L}_{\mathcal{B}}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left( f_\theta(x_i) - y_i \right)^2
$$

Update at iteration $n$, with minibatch $\mathcal{B}^{(n)}$:

$$
\theta^{(n+1)} = \theta^{(n)} + \Delta\theta^{(n)},
\qquad
\Delta\theta^{(n)} = G\left(\theta^{(n)}, \nabla_\theta \mathcal{L}_{\mathcal{B}^{(n)}}(\theta^{(n)}), s^{(n)}\right)
$$

- $G$: optimizer rule.
- $s^{(n)}$: optimizer state, e.g. past gradients.






## Where $M$ and $d(M, \tilde M)$ are stored

- **Maps**: $M_{p,t}$ is the column `tmCO2 [kg]` of the file
  `spe11b/<p>/spe11b_spatial_map_<t>y.csv`, for $t \in \{0, 5, \dots, 1000\}$.
- **Distances**: $d(M_{p,t}, M_{\tilde p,\tilde t})$ is the column `distance` of
  `spe11b/assembled_sorted_full.csv`, in the row with `key1` $=$ `<p>_<t>y` and
  `key2` $=$ `<p̃>_<t̃>y`, for all $p, \tilde p$ and $t, \tilde t \in \{50, 100, \dots, 1000\}$,
  including $t \neq \tilde t$. 
- Key notation: `<·>` is a placeholder; `_` and `y` (years) are literal, e.g. `calgary1_100y`.


## Data Management Implementation



**Indexing:**

- $\mathcal{P} = \{\text{calgary1}, \ldots, \text{ut-csee2}\}$: the names of the 34 participants, in alphabetical order.
- $a \in \{0, \dots, |\mathcal{P}|-1\}$: participant index, the position of $p$ in $\mathcal{P}$.
- $\mathcal{T} = \{50, 100, \dots, 1000\}$: the years with distances, in increasing order.
- $b \in \{0, \dots, |\mathcal{T}|-1\}$: year index, the position of $t$ in $\mathcal{T}$.
- $K = |\mathcal{P}| \cdot |\mathcal{T}| = 680$.
- $k \in \{0, \dots, K-1\}$: unrolled index, one per map; the participant index moves faster (or the opposite, with $\Lambda$ and $\Lambda^{-1}$ below changed accordingly).


**From training index to coupled unrolled index:**


$$
\pi : \{0, \dots, N-1\} \to \{0, \dots, K-1\}^2,
\qquad
\pi(i) = (k_i, \tilde k_i)
$$



**From unrolled index to label:**

- $\lambda = (p, t) \in \mathcal{P} \times \mathcal{T}$: label of a map.

$$
\Lambda : \{0, \dots, K-1\} \to \mathcal{P} \times \mathcal{T},
\qquad
\Lambda(k) = (p, t),
\quad \text{with} \quad
a = k \bmod |\mathcal{P}|,
\quad
b = \lfloor k / |\mathcal{P}| \rfloor
$$

- E.g., $\Lambda(k) = (\texttt{participant}[k], \texttt{year}[k])$; $\Lambda(0) = (\texttt{calgary1}, 50)$, $\Lambda(1) = (\texttt{cau-kiel1}, 50)$. (This uses the assumption that names, not indices, are kept.)
- Inverse, from label to unrolled index: $\Lambda^{-1}(p, t) = |\mathcal{P}| \, b + a$. Preprocessing uses it to fill `distances` from the CSV keys.




**Preprocessing, run once:**

$$
\text{preprocess} : \texttt{spe11b/} \mapsto \texttt{HDF5 file}
$$

- The HDF5 file is stored in `data/`. It is a second starting point: training reads only `data/`, not `spe11b/`.

Content of `data/`: one HDF5 file, `data/spe11b.h5`, all unscaled. `data/spe11b.h5` is everything the user needs, alongside the provided utilities to select specific data from it for training. Why HDF5 format? Because it is common and relatively easy to use. 


Content of the HDF5 file:
| Dataset | Shape | Type | Content |
|---|---|---|---|
| `maps` | $(K, 120, 840)$ | float32 | $M$, in kg |
| `distances` | $(K, K)$ | float64 | $d(M, \tilde M)$, in kg·m |
| `participant` | $(K,)$ | string | participant $p \in \mathcal{P}$ of each map, e.g. `calgary1` |
| `year` | $(K,)$ | int | year $t \in \mathcal{T}$ of each map, e.g. 50 |



Redundancy:
 
- `distances` is symmetric, and labels repeat. Estimated 2 MB in total: accepted.



**Data management utilities:**

- ML needs fast minibatch creation.
- The HDF5 file is kept in memory.
- Data is picked on the fly, when a minibatch is built.
- Required: a function, $g$, from a training index to a scaled training sample:

$$
g : \{0, \dots, N-1\} \to \left( \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \right) \times \mathbb{R},
\qquad
g : i \mapsto \left( \left( \mathcal{S}_M(M_{\lambda_i}), \mathcal{S}_M(M_{\tilde\lambda_i}) \right), \mathcal{S}_d\left( d(M_{\lambda_i}, M_{\tilde\lambda_i}) \right) \right)
$$


- $\phi$: from a label pair to a training sample, with $M_{\lambda} = M_{p,t}$ for $\lambda = (p, t)$:

$$
\phi : (\mathcal{P} \times \mathcal{T})^2 \to \left( \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \right) \times \mathbb{R}_{\ge 0},
\qquad
\phi : (\lambda, \tilde\lambda) \mapsto \left( (M_{\lambda}, M_{\tilde\lambda}), d(M_{\lambda}, M_{\tilde\lambda}) \right)
$$

- We have:

$$
g = (\mathcal{S}_M \times \mathcal{S}_M \times \mathcal{S}_d) \circ \phi \circ (\Lambda \times \Lambda)  \circ \pi,
$$


**Coding:**

The functions $\pi$, $\Lambda$, $\phi$, $\mathcal{S}_M$, $\mathcal{S}_d$ composing $g$  are provided in `src/spe11_wasserstein/*.py`.




## PyTorch users
- `__getitem__` corresponds to $g$: inside `__getitem__`, call the functions that compose $g$.

## JAX users
- The dataset is typically custom, so implementing $g$ is even more straightforward.

