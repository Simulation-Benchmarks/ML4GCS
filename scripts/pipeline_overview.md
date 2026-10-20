## Assumptions

- Python, with common libraries (NumPy, JAX, PyTorch, optax, pandas), for interoperability.
- Memory is limited: each map is stored once, instead of storing every input pair explicitly, as is common practice. This requires a preprocessing step and data-management utilities.
- A pair is a pair of indices, never a copy of two maps, to save memory.
- `spe11b/` holds the raw data and is never modified: this implementation is a layer on top of it, for modularity.
- Different ML users may use the data differently for training: the data-management logic must be flexible and modular.
- Participants and years are represented by integer indices (see Indexing), not by names (e.g. `calgary1`) or year values (e.g. 50), so that JAX users can jit the whole training.


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

Let $\mathcal{D} = \{ \text{datum}_q : q = 0, \dots, |\mathcal{D}|-1 \}$ be the full dataset: $\text{datum}_q$ is the sample with datum index $q$.

- Each sample is $\left( (M, \tilde M), d(M, \tilde M) \right)$: two maps and the distance between them.
- One sample per ordered pair of maps, including $\tilde M = M$: $(M, \tilde M)$ and $(\tilde M, M)$ are different samples.
- $|\mathcal{D}| = 680^2 = 462{,}400$: 680 maps, from 34 participants and 20 years.

The split is a random partition of the datum indices $q$, in the ratio $0.7 : 0.15 : 0.15$. The three parts select the samples of the training dataset $\mathcal{D}_{\text{train}}$, the validation dataset $\mathcal{D}_{\text{val}}$ and the test dataset $\mathcal{D}_{\text{test}}$.

- $N = |\mathcal{D}_{\text{train}}| = 323{,}680$: number of training samples. $|\mathcal{D}_{\text{val}}| = |\mathcal{D}_{\text{test}}| = 69{,}360$.
- $i \in \{0, \dots, N-1\}$: training index, identifies one sample of $\mathcal{D}_{\text{train}}$.



## Scaling
- Scalings: $\mathcal{S}_M$ for maps, $\mathcal{S}_d$ for distances. Both invertible.

For example, standardization (as in `case_3_cnn_oop`):

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
- $\mathcal{I} = (I_0, \dots, I_{N-1})$: training index sequence, initially $(0, \dots, N-1)$.
- $\text{reshuffle}(\mathcal{I})$: the entries of $\mathcal{I}$ in a uniformly random order.

Each epoch:

1. If the user enables reshuffle: $\mathcal{I} \leftarrow \text{reshuffle}(\mathcal{I})$.
2. For $j = 0, \dots, J-1$: run iteration $n$ with $\mathcal{B}^{(n)}$ below, then $n \leftarrow n + 1$.

$$
\mathcal{B}^{(n)} = \left\{ I_m : j \, B \le m < \min\left( (j+1) \, B, N \right) \right\}
$$

Each epoch uses every training index exactly once.



## Typical training
- Training index: $i \in \{0, \dots, N-1\}$.
- $\lambda = (a, b)$: label of a map; $a$, $b$ are the participant and year indices (see Indexing). In particular, 
 $\lambda_i = (a_i, b_i)$, $\tilde\lambda_i = (\tilde a_i, \tilde b_i)$: labels of the two maps of training sample $i$.
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
\left( \Delta\theta^{(n)}, s^{(n+1)} \right) = G\left(\theta^{(n)}, \nabla_\theta \mathcal{L}_{\mathcal{B}^{(n)}}(\theta^{(n)}), s^{(n)}\right)
$$

- $G$: optimizer rule.
- $s^{(n)}$: optimizer state, e.g. past gradients.






## Where $M$ and $d(M, \tilde M)$ are currently stored

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
- $a \mapsto p$ and $b \mapsto t$ are bijections, stored as two arrays of the HDF5 file: $p = \texttt{participants}[a]$, $t = \texttt{years}[b]$. Inverses: a dictionary, e.g. `{p: a for a, p in enumerate(participants)}`.
- $K = |\mathcal{P}| \cdot |\mathcal{T}| = 680$.
- $k \in \{0, \dots, K-1\}$: unrolled index, one per map; the participant index moves faster (or the opposite, with $\Lambda$ and $\Lambda^{-1}$ below changed accordingly).


**From training index to pair of unrolled indices:**


$$
\pi : \{0, \dots, N-1\} \to \{0, \dots, K-1\}^2,
\qquad
\pi(i) = (k_i, \tilde k_i)
$$

- $\pi(i) = \left( \lfloor q_i / K \rfloor, \, q_i \bmod K \right)$.
- $q = K k + \tilde k$: datum index of the sample with unrolled indices $(k, \tilde k)$.
- $(q_0, \dots, q_{N-1})$: the datum indices of $\mathcal{D}_{\text{train}}$, in random order.


**From unrolled index to label:**


$$
\begin{gathered}
\Lambda : \{0, \dots, K-1\} \to \{0, \dots, |\mathcal{P}|-1\} \times \{0, \dots, |\mathcal{T}|-1\}, \\
\lambda = \Lambda(k) = (a, b),
\quad \text{with} \quad
a = k \bmod |\mathcal{P}|,
\quad
b = \lfloor k / |\mathcal{P}| \rfloor
\end{gathered}
$$

- E.g., $\Lambda(0) = (0, 0)$, label of $M_{\text{calgary1}, 50}$; $\Lambda(1) = (1, 0)$, label of $M_{\text{cau-kiel1}, 50}$.
- Inverse, from label to unrolled index: $\Lambda^{-1}(a, b) = |\mathcal{P}| \, b + a$. Preprocessing uses it to fill `distances` from the CSV keys.


**Schematically:**

$$
\begin{array}{cccccccccccccl}
 & & & & & & i & & & & & & & \qquad \text{training index} \\
 & & & & \swarrow & & & & \searrow & & & & & \qquad \pi \\
 & & k_i & & & & & & & & \tilde k_i & & & \qquad \text{unrolled indices} \\
 & \swarrow & & \searrow & & & & & & \swarrow & & \searrow & & \qquad \Lambda \\
 a_i & & & & b_i & & & & \tilde a_i & & & & \tilde b_i & \qquad \text{participant and year indices}
\end{array}
$$




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
| `distances` | $(K, K)$ | float32 | $d(M, \tilde M)$, in kg·m |
| `participants` | $(\lvert\mathcal{P}\rvert,)$ | string | $\mathcal{P}$, e.g. `participants[0]` = `calgary1` |
| `years` | $(\lvert\mathcal{T}\rvert,)$ | int | $\mathcal{T}$, e.g. `years[0]` = 50 |



Redundancy:
 
- `distances` is symmetric. Estimated 1 MB in total: accepted.



**Data management utilities:**

- ML needs fast minibatch creation.
- The arrays of the HDF5 file are kept in GPU memory (≈ 276 MB).
- Data is picked on the fly, when a minibatch is built.
- Required: a function, $g$, from a training index to a scaled training sample:

$$
\begin{gathered}
g : \{0, \dots, N-1\} \to \left( \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \right) \times \mathbb{R}, \\
g : i \mapsto \left( \left( \mathcal{S}_M(M_{\lambda_i}), \mathcal{S}_M(M_{\tilde\lambda_i}) \right), \mathcal{S}_d\left( d(M_{\lambda_i}, M_{\tilde\lambda_i}) \right) \right)
\end{gathered}
$$


- $\phi$: from a label pair to a training sample, with $M_{\lambda} = M_{p,t}$ for $\lambda = (a, b)$, where $a$, $b$ are the indices of $p$, $t$:

$$
\begin{gathered}
\phi : \left( \{0, \dots, |\mathcal{P}|-1\} \times \{0, \dots, |\mathcal{T}|-1\} \right)^2 \to \left( \mathbb{R}^{120 \times 840} \times \mathbb{R}^{120 \times 840} \right) \times \mathbb{R}_{\ge 0}, \\
\phi : (\lambda, \tilde\lambda) \mapsto \left( (M_{\lambda}, M_{\tilde\lambda}), d(M_{\lambda}, M_{\tilde\lambda}) \right)
\end{gathered}
$$

- We have:

$$
g = ((\mathcal{S}_M \times \mathcal{S}_M) \times \mathcal{S}_d) \circ \phi \circ (\Lambda \times \Lambda)  \circ \pi
$$


**Coding:**

The functions $\pi$, $\Lambda$, $\phi$, $\mathcal{S}_M$, $\mathcal{S}_d$ composing $g$  are provided in `src/spe11_wasserstein/*.py`.

- One dataset instance per split ($\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test}}$): same arrays in memory, a separate $\pi$, the same $\mathcal{S}_M$, $\mathcal{S}_d$ as $\mathcal{D}_{\text{train}}$.




## PyTorch users
- `__getitem__` corresponds to $g$: inside `__getitem__`, call the functions that compose $g$.
- `__len__` returns $N$; `DataLoader` needs it.
- `DataLoader(dataset, batch_size=B, shuffle=True)` implements the Minibatching section: `shuffle` enables the reshuffle; the last minibatch has $N - (J-1) \, B$ samples (default `drop_last=False`).
- Keep the default `num_workers=0`: the arrays are CUDA tensors, which do not work reliably with worker processes.


## JAX users
- The dataset is typically custom, so implementing $g$ is even simpler.
- No built-in data loader: write the epoch loop of the Minibatching section.
- $g$ uses integer indices only: jit it together with the training step, with `maps`, `distances` and the pairs $\pi(i)$ on the GPU.



## Utilities
- `find_spe11b_data_root(start="spe11b")` in `src/spe11_wasserstein/discovery.py`: returns the folder that directly contains the participant folders.

