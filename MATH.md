# The Mathematics of Heterogeneous Newton Boosting Machines

This note derives the Heterogeneous Newton Boosting Machine (HNBM), explains
the mathematics implemented by this repository, develops the included NNBoost
realization, and compares HNBM with traditional gradient boosting, Newton tree
boosting, SnapBoost, and XGBoost.

## 1. Problem setup

Let the training data be

$$
\mathcal D=\{(x_i,y_i,w_i)\}_{i=1}^{n},
\qquad w_i\geq0,
$$

and define empirical risk

$$
\mathcal R(F)=\sum_{i=1}^{n}w_i\ell(y_i,F(x_i)).
$$

The goal is to learn a function $F$ that minimizes this empirical risk. HNBM
represents it as an additive ensemble

$$
F_M(x)=F_0+\sum_{m=1}^{M}\eta_m f_m(x).
$$

Unlike a homogeneous booster, HNBM has a pool of hypothesis subclasses

$$
\mathbb{H}=\{\mathcal H_1,\ldots,\mathcal H_K\}
$$

and a probability vector

$$
p=(p_1,\ldots,p_K),
\qquad p_k\geq0,
\qquad \sum_{k=1}^{K}p_k=1.
$$

A subclass might contain trees of a given depth, networks of a given width,
kernel regressors of a given bandwidth, linear models, or another family whose
members can be fit as weighted regressors.

## 2. Functional gradient boosting

For a small functional perturbation $f$, first-order boosting uses

$$
\mathcal R(F+f)\approx\mathcal R(F)
+\sum_iw_i g_i f(x_i),
$$

where

$$
g_i=\left.\frac{\partial\ell(y_i,z)}{\partial z}
\right|_{z=F(x_i)}.
$$

The negative functional gradient is represented on the training observations
by the pseudo-residuals

$$
u_i=-g_i.
$$

A standard gradient boosting round fits a base learner to $(x_i,u_i)$ and
updates $F$. This accounts for the slope of the objective but ignores how
quickly the slope changes.

## 3. Newton boosting as weighted least squares

### 3.1 Second-order expansion

For a proposed correction $q_i=f(x_i)$, expand each loss around the current
score $F_{m-1}(x_i)$:

$$
\ell(y_i,F_{m-1}(x_i)+q_i)
\approx\ell_i+g_iq_i+\frac12h_iq_i^2,
$$

where

$$
h_i=\left.\frac{\partial^2\ell(y_i,z)}{\partial z^2}
\right|_{z=F_{m-1}(x_i)}.
$$

When $h_i>0$, completing the square gives

$$
g_iq_i+\frac12h_iq_i^2
=\frac12h_i\left(q_i+\frac{g_i}{h_i}\right)^2
-\frac{g_i^2}{2h_i}.
$$

The final term is independent of $q_i$. Minimizing the quadratic surrogate
over a hypothesis subclass is therefore equivalent to

$$
f_{m,k}\in\arg\min_{f\in\mathcal H_k}
\sum_{i=1}^{n}w_i h_i\left(r_i-f(x_i)\right)^2,
\qquad
r_i=-\frac{g_i}{h_i}.
$$

This weighted least-squares problem is HNBM's common interface between
optimization and arbitrary base learners:

- $r_i=-g_i/h_i$ is the Newton working response;
- $\widetilde w_i=w_i h_i$ is the effective fitting weight; and
- the learner approximates, or projects, the unrestricted Newton step within
  its own function class.

The repository implements these quantities directly as
`newton_target = -g / h` and
`fit_weight = h * sample_weight`.

### 3.2 Ensemble update and shrinkage

After fitting a learner, the ensemble becomes

$$
F_m(x)=F_{m-1}(x)+\eta_mf_m(x).
$$

The default uses fixed shrinkage $\eta_m=\eta$. With `line_search=True`, this
implementation evaluates

$$
\eta_m\in\eta\{0.25,0.5,1,1.5,2\}
$$

and chooses the value with the smallest current training loss. This is a
finite grid search rather than an exact continuous line search.

### 3.3 Why both response and weight are needed

It may appear sufficient to fit $-g_i/h_i$ without weights, but that would not
minimize the Taylor surrogate. Observations with larger curvature have greater
effect on local objective change, which is exactly represented by $h_i$ in the
weighted regression objective. Multiplying by the original observation weight
gives $w_i h_i$ without changing the pointwise Newton target.

## 4. Objectives in this implementation

### 4.1 Squared-error regression

HNBM defines

$$
\ell(y,F)=(F-y)^2,
\qquad g=2(F-y),
\qquad h=2.
$$

Hence

$$
r=-\frac gh=y-F.
$$

Newton boosting and ordinary residual boosting coincide because the curvature
is constant. The initial score is the weighted mean

$$
F_0=\frac{\sum_iw_i y_i}{\sum_iw_i}.
$$

### 4.2 Binary logistic classification

The two observed classes are mapped internally to $y\in\{-1,+1\}$. The loss
is

$$
\ell(y,F)=\log(1+e^{-yF}).
$$

Let $a=\sigma(-yF)$ with $\sigma(t)=1/(1+e^{-t})$. Then

$$
g=-ya,
\qquad
h=a(1-a)=\sigma(-yF)\sigma(yF),
$$

and

$$
r=-\frac gh=\frac{y}{1-a}=\frac{y}{\sigma(yF)}.
$$

The Hessian is floored at machine epsilon for numerical stability. If

$$
\widehat p=\frac{\sum_iw_i\mathbf{1}_{\{y_i=+1\}}}{\sum_iw_i},
$$

the initial raw score is the clipped empirical log-odds

$$
F_0=\log\frac{\widehat p}{1-\widehat p}.
$$

The positive-class probability is

$$
P(y=+1\mid x)=\sigma(F_M(x)).
$$

### 4.3 Pseudo-Huber regression

For residual $e=F-y$ and scale $\delta>0$,

$$
\ell_\delta(y,F)=\delta^2
\left(\sqrt{1+(e/\delta)^2}-1\right),
$$

$$
g=\frac{e}{\sqrt{1+(e/\delta)^2}},
\qquad
h=\left(1+(e/\delta)^2\right)^{-3/2}.
$$

Large residuals have low curvature and therefore low effective fitting weight,
providing smooth robustness to outliers. The implementation floors $h$ at
machine epsilon and uses the weighted mean as $F_0$.

### 4.4 Quantile regression

For $\tau\in(0,1)$ and residual $e=y-F$, the pinball loss is

$$
\rho_\tau(e)=\max\{\tau e,(\tau-1)e\}.
$$

Because pinball loss is not twice differentiable, this implementation uses

$$
g=1-\tau, \qquad F\geq y,
$$

$$
g=-\tau, \qquad F<y,
\qquad h=1.
$$

as a unit-Hessian working approximation. Thus the shared fitting mechanism is
used, but the update is not a literal Newton step for pinball loss. The initial
score is the weighted empirical $\tau$-quantile.

## 5. Heterogeneous learner selection

### 5.1 Random HNBM

The default algorithm samples

$$
K_m\sim\mathrm{Categorical}(p_1,\ldots,p_K),
$$

then fits only

$$
f_m=f_{m,K_m}
$$

using the Newton weighted-regression objective. Random subclass selection
reduces per-round work when learners are expensive and diversifies the sequence
of functional corrections. The
probabilities encode how frequently each inductive bias receives an
opportunity to approximate the Newton direction.

If all $\mathcal H_k$ are identical, HNBM reduces to a randomized realization
of homogeneous Newton boosting. If $K=1$, it reduces to ordinary Newton
boosting over that one class.

### 5.2 Greedy selection

With `selection_strategy="greedy"`, every positive-probability candidate is
fit to the same working problem. HNBM selects

$$
(k_m,\eta_m)\in\arg\min_{k,\eta_k}
\sum_iw_i\ell\left(y_i,
F_{m-1}(x_i)+\eta_k f_{m,k}(x_i)\right).
$$

Candidate fitting may run in parallel. In this mode, probabilities determine
which candidates are eligible ($p_k>0$), but their magnitudes do not enter the
loss comparison.

### 5.3 Row subsampling

When `subsample=s<1`, HNBM draws without replacement

$$
n_s=\left\lceil s n_+\right\rceil
$$

eligible observations, where $n_+$ is the number having positive effective
weight $w_i h_i$. The learner is fit on that sample, while its predictions and
greedy-selection loss are evaluated on the complete training set.

### 5.4 Early stopping

For validation data, HNBM records

$$
\mathcal R_{\mathrm{val}}^{(m)}
=\frac1{n_{\mathrm{val}}}\sum_i
\ell(y_i^{\mathrm{val}},F_m(x_i^{\mathrm{val}})).
$$

If improvement greater than `min_delta` does not occur for
`early_stopping_rounds`, the fitted learner list is truncated back to the best
recorded ensemble size.

## 6. NNBoost: HNBM with shallow neural learners

NNBoost is the ready-to-use HNBM realization included in this package. Its
subclasses differ by hidden width:

$$
\mathbb{H}=\{\mathcal H_{q_1},\ldots,\mathcal H_{q_K}\},
$$

where the default widths are $(16,32,64)$ and

$$
p_k=\frac1K.
$$

### 6.1 Network function

Inputs are standardized using effective sample weights. For standardized
$\widetilde x\in\mathbb{R}^d$, a width-$q$ learner predicts a standardized
working response with

$$
\widetilde f_\theta(x)
=v^\top a(W^\top\widetilde x+b)+c,
$$

where

- $W\in\mathbb{R}^{d\times q}$ and $b\in\mathbb{R}^q$ are hidden-layer
  parameters;
- $v\in\mathbb{R}^q$ and $c\in\mathbb{R}$ are output parameters; and
- $a$ is ReLU, tanh, or logistic activation.

The Newton responses are also standardized using their effective weighted mean
$\mu_r$ and standard deviation $s_r$:

$$
\widetilde r_i=\frac{r_i-\mu_r}{s_r}.
$$

After training, the correction returned to HNBM is

$$
f_\theta(x)=s_r\widetilde f_\theta(x)+\mu_r.
$$

### 6.2 Base-network objective

Let $\widetilde w_i=w_i h_i$. The inner learner minimizes

$$
J(\theta)=
\frac{\sum_i\widetilde w_i
\left(\widetilde f_\theta(x_i)-\widetilde r_i\right)^2}
{\sum_i\widetilde w_i}
+\alpha\left(\lVert W\rVert_F^2+\lVert v\rVert_2^2\right).
$$

Bias terms are not penalized. The learner applies full-batch gradient descent
with step `learning_rate_nn`, clips the joint gradient norm to at most 5, and
stops after `max_iter` epochs or once consecutive objective values differ by
less than `tol`.

For example, with error
$e_i=\widetilde f_\theta(x_i)-\widetilde r_i$ and total weight
$S=\sum_i\widetilde w_i$, the output-layer gradients include

$$
\nabla_vJ=\sum_i\frac{2\widetilde w_i e_i}{S}z_i+2\alpha v,
\qquad
\frac{\partial J}{\partial c}=\sum_i\frac{2\widetilde w_i e_i}{S},
$$

where $z_i=a(W^\top\widetilde x_i+b)$. Backpropagation propagates
$2\widetilde w_i e_i/S$ through $v$ and $a'$ to obtain gradients for $W$ and
$b$.

### 6.3 Two distinct learning rates

NNBoost has two learning rates with different roles:

$$
\theta^{(t+1)}=\theta^{(t)}
-\eta_{\mathrm{NN}}\nabla_\theta J(\theta^{(t)})
$$

inside each base network, and

$$
F_m=F_{m-1}+\eta_m f_m
$$

between boosting rounds. In the API these are `learning_rate_nn` and
`learning_rate`, respectively. Confusing them changes different levels of the
optimization.

## 7. Complete HNBM algorithm

The implementation can be summarized as follows:

1. Validate the learner pool and probabilities, map binary labels to
   $\{-1,+1\}$ if needed, and compute $F_0$.
2. For $m=1,\ldots,M$:
   1. Compute $g_i$ and $h_i$ at $F_{m-1}(x_i)$.
   2. Form $r_i=-g_i/h_i$ and $\widetilde w_i=w_i h_i$.
   3. Sample $K_m$ from the configured categorical distribution, or fit all
      eligible candidates and compare their updated losses.
   4. Optionally subsample positive-weight observations.
   5. Fit the selected cloneable regressor to $r_i$ with weights
      $\widetilde w_i$.
   6. Use fixed shrinkage or choose $\eta_m$ from the discrete line-search grid.
   7. Add the scaled learner to $F_{m-1}$, record losses and metrics, and invoke
      callbacks.
3. Optionally restore the best validation ensemble.

Prediction uses exactly

$$
F_{M'}(x)=F_0+\sum_{m=1}^{M'}\eta_mf_m(x),
$$

where $M'\leq M$ after early stopping or explicit callback termination.

## 8. Comparison with traditional boosting methods

### 8.1 AdaBoost

Classical AdaBoost minimizes exponential loss

$$
\ell(y,F)=e^{-yF}
$$

through iterative reweighting and a weighted vote of classifiers. HNBM instead
fits real-valued Newton corrections for differentiable objectives. Observation
weights are combined with Hessian weights rather than updated according to a
specific exponential-loss rule.

### 8.2 First-order gradient boosting

First-order gradient boosting fits $-g_i$. Newton boosting fits $-g_i/h_i$ with
weight $w_i h_i$. They are equivalent up to constant scaling for squared error,
but differ when curvature varies across observations, such as logistic or
pseudo-Huber loss.

### 8.3 Homogeneous Newton boosting

Traditional Newton boosting normally uses one hypothesis class $\mathcal H$ at
every round:

$$
f_m\in\arg\min_{f\in\mathcal H}
\sum_iw_i h_i(r_i-f(x_i))^2.
$$

HNBM replaces $\mathcal H$ with a sampled or greedily chosen
$\mathcal H_{K_m}$. The derivative calculation stays unchanged, but each class
projects the Newton direction using a different inductive bias.

### 8.4 SnapBoost

SnapBoost is a concrete HNBM whose pool combines tree subclasses and random-
Fourier-feature ridge subclasses, with an optional raw linear learner. NNBoost
instead uses shallow-network widths as its subclasses. In both cases, the
outer algorithm is the same Newton regression, learner-selection, and additive
update procedure; only the learner pool and its probability allocation differ.

## 9. Detailed comparison with XGBoost

XGBoost also uses a second-order approximation. For a proposed tree $f_t$,

$$
\widetilde{\mathcal L}^{(t)}
=\sum_i\left[g_i f_t(x_i)+\frac12h_i f_t(x_i)^2\right]
+\Omega(f_t),
$$

with a common tree regularizer

$$
\Omega(f)=\gamma_T T+\frac12\lambda\sum_{j=1}^{T}c_j^2.
$$

For a fixed tree partition $R_1,\ldots,R_T$, define

$$
G_j=\sum_{i\in R_j}g_i,
\qquad H_j=\sum_{i\in R_j}h_i.
$$

The optimal XGBoost leaf value is

$$
c_j^*=-\frac{G_j}{H_j+\lambda},
$$

and the canonical gain from splitting a parent into left and right children is

$$
\mathrm{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}
+\frac{G_R^2}{H_R+\lambda}
-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma_T.
$$

The shared second-order foundation is clear, but the minimization strategy is
different:

| Aspect | HNBM | XGBoost tree booster |
|---|---|---|
| Quadratic information | Newton responses and Hessian sample weights | Aggregated gradient and Hessian statistics |
| Learner class | Any configured weighted-regressor pool | Regularized regression trees |
| Per-round choice | Random subclass by default; optional greedy selection | Greedy or approximate split search within the tree class |
| Regularization | Supplied by each learner plus shrinkage, sampling, and early stopping | Explicit leaf and structure penalties plus tree constraints |
| Leaf formula | Depends on the configured tree estimator | Analytic regularized value $-G_j/(H_j+\lambda)$ |
| Neural corrections | Supported through NNBoost | Not part of the standard tree booster |
| Heterogeneity | Different classes can coexist in one ensemble | Standard booster repeatedly uses trees |

For a fixed unregularized tree partition, the weighted Newton regression gives

$$
c_j=-\frac{\sum_{i\in R_j}w_i g_i}
{\sum_{i\in R_j}w_i h_i},
$$

which is the weighted $\lambda=0$ analogue of XGBoost's regularized leaf
formula. XGBoost incorporates the regularizer directly in leaf optimization
and split scoring. HNBM delegates regularization and function fitting to
whichever learner was selected.

## 10. Convergence interpretation

At each iteration, HNBM does not generally compute the exact unrestricted
Newton step. It computes a class-constrained approximation. A useful abstract
quality measure for class $k$ is how much of a target direction $r$ it can
capture under the Hessian-weighted norm

$$
\lVert u\rVert_H^2=\sum_iw_i h_i u_i^2.
$$

If $P_k r$ denotes the best projection of $r$ onto the predictions available
from $\mathcal H_k$, one may describe its relative approximation quality by

$$
\rho_k(r)=\frac{\lVert P_k r\rVert_H^2}{\lVert r\rVert_H^2},
\qquad 0\leq\rho_k(r)\leq1.
$$

Under strong-convexity, smoothness, and weak-learning assumptions, a positive
expected capture

$$
\mathbb{E}_{K\sim p}[\rho_K(r)]
=\sum_kp_k\rho_k(r)>0
$$

supports a geometric decrease in an appropriate optimization error measure.
This expression also gives intuition for learner probabilities: assigning more
mass to classes that reliably approximate remaining Newton directions can
improve expected progress, while retaining diverse classes can cover directions
that a single family represents poorly. Exact convergence constants depend on
the formal assumptions and HNBM variant; this section is an interpretation,
not a substitute for the theorem in the SnapBoost/HNBM literature.

## 11. Regularization and computational tradeoffs

HNBM exposes regularization at several levels:

- **Boosting shrinkage** controls the outer step $\eta_m$.
- **Learner regularization** is specific to each class; NNBoost uses L2 weight
  decay, hidden width, and inner early stopping.
- **Learner probabilities** allocate expected compute and representation
  opportunities among classes.
- **Row subsampling** reduces per-round fitting cost and injects randomness.
- **Greedy selection** spends more compute to compare immediate loss reduction.
- **Validation early stopping** selects effective ensemble length.
- **Compaction** may remove learners with sufficiently small stored
  contribution weights after training.

For random selection, expected fitting cost per round is approximately

$$
\mathbb{E}[C]=\sum_{k=1}^{K}p_k C_k,
$$

where $C_k$ is the cost of fitting class $k$. Greedy selection instead costs
roughly $\sum_{k:p_k>0}C_k$ per round, reduced in wall-clock time when candidate
fits run in parallel.

## 12. Implementation-specific summary

The following details describe this repository rather than every possible
HNBM implementation:

- Base learners must be cloneable regressors whose `fit` explicitly accepts
  `sample_weight`.
- Classification is binary and uses logistic loss.
- Regression supports squared error, pseudo-Huber, and quantile objectives;
  quantile uses the unit-Hessian approximation described above.
- Random learner selection is the default; greedy selection and the discrete
  line-search grid are optional.
- NNBoost's default pool contains widths 16, 32, and 64 with equal probability.
- Each fitted neural learner standardizes features and working targets using
  effective weights, then optimizes weighted MSE by full-batch gradient descent.
- Validation history, callbacks, subsampling, early stopping, and post-fit
  compaction are orchestration features around the same Newton core.

## References

- Thomas Parnell et al., *SnapBoost: A Heterogeneous Boosting Machine*, NeurIPS
  2020, [arXiv:2006.09745](https://arxiv.org/abs/2006.09745).
- Jerome H. Friedman, *Greedy Function Approximation: A Gradient Boosting
  Machine*, Annals of Statistics, 2001.
- Tianqi Chen and Carlos Guestrin, *XGBoost: A Scalable Tree Boosting System*,
  KDD, 2016.
- Michael Sigrist, *Gradient and Newton Boosting for Classification and
  Regression*, 2018.
