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

$F(x)$ is a scalar for regression and binary classification. For multiclass
classification it is a $K$-vector of class scores, and each $f_m$ is a stack
of $K$ scalar learners from the same hypothesis class.

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
`fit_weight = h * sample_weight`. When $F$ is vector-valued, as in
multiclass softmax, the same completing-the-square argument is applied
coordinate-wise after replacing the full Hessian with its diagonal; see
§4.2.1.

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

### 4.2.1 Multiclass softmax classification

For $K>2$ observed classes the raw score is a vector $F(x)\in\mathbb{R}^K$.
Integer labels are mapped internally to codes $y\in\{0,\ldots,K-1\}$. Softmax
probabilities are

$$
p_k=\mathrm{softmax}(F)_k
=\frac{e^{F_k}}{\sum_{j=1}^{K}e^{F_j}},
\qquad k=1,\ldots,K.
$$

(The implementation indexes classes from $0$ through $K-1$; the mathematics is
the same.) With a one-hot vector $y\in\{0,1\}^K$ the multinomial loss is

$$
\ell(y,F)=-\sum_{k=1}^{K}y_k\log p_k=-\log p_{y}.
$$

Equivalently, with the log-partition function $Z=\sum_j e^{F_j}$,

$$
\ell(y,F)=\log Z-F_{y},
\qquad
\nabla_F\log Z=p,
\qquad
\nabla_F^2\log Z=H.
$$

Softmax is translation-invariant,
$\mathrm{softmax}(F+c\mathbf{1})=\mathrm{softmax}(F)$. This repository keeps
all $K$ scores, matching XGBoost and LightGBM, rather than reducing to $K-1$
free coordinates.

Writing $Z=\sum_j e^{F_j}$, the softmax Jacobian is

$$
\frac{\partial p_k}{\partial F_j}=p_k(\delta_{kj}-p_j).
$$

The chain rule on $\ell=-\log p_{y}$ is

$$
\frac{\partial\ell}{\partial F_k}
=-\frac1{p_{y}}\frac{\partial p_{y}}{\partial F_k}
=-(\delta_{yk}-p_k)
=p_k-\mathbf{1}_{\{k=y\}},
$$

and a second derivative produces $H_{kj}=p_k(\delta_{kj}-p_j)$ as below. For a
vector correction $q\in\mathbb{R}^K$ the second-order expansion is

$$
\ell(y,F+q)
\approx\ell(y,F)+g^\top q+\frac12 q^\top H q.
$$

The exact Hessian has the form $H=\mathrm{diag}(p)-pp^\top$. If $e$ is a
one-hot draw from $\mathrm{Categorical}(p)$, then $H=\mathrm{Cov}(e)$,
$H_{kk}=\mathrm{Var}(e_k)=p_k(1-p_k)$, and $H_{kj}=-\,p_kp_j$ for $k\neq j$.
Hence

$$
H\mathbf{1}=0,
\qquad \mathrm{rank}(H)=K-1,
\qquad H\succeq0,
\qquad \ker H=\mathrm{span}\{\mathbf{1}\}.
$$

The gradient is orthogonal to the same direction,

$$
\mathbf{1}^\top g=\sum_k(p_k-\mathbf{1}_{\{k=y\}})=0,
$$

so $g\in\mathrm{range}(H)$ and a full Newton system $Hq=-g$ is consistent but
singular without a gauge constraint on $q$. Differentiating with respect to the
scores gives the gradient

$$
g_k=\frac{\partial\ell}{\partial F_k}=p_k-\mathbf{1}_{\{k=y\}}
$$

and the exact per-observation Hessian

$$
H_{kj}=\frac{\partial^2\ell}{\partial F_k\partial F_j}
=p_k\bigl(\delta_{kj}-p_j\bigr).
$$

A full Newton step would invert this $K\times K$ matrix at every sample. The
base learners here are scalar regressors, so the implementation keeps only the
diagonal,

$$
h_k=H_{kk}=p_k(1-p_k),
\qquad
h_k\leftarrow\max(h_k,\varepsilon).
$$

XGBoost and LightGBM take the same diagonal but inflate it by a constant factor
greater than one before dividing: XGBoost uses $2p_k(1-p_k)$ and LightGBM uses
$\tfrac{K}{K-1}p_k(1-p_k)$. That factor is damping rather than a derivative.
The bare diagonal is not an upper bound on $H$, because

$$
\mathrm{diag}(h)-H=pp^\top-\mathrm{diag}(p)^2
$$

has a zero diagonal and non-negative off-diagonal entries, so it is indefinite
whenever at least two classes carry mass. Inflating $h$ buys a margin against
that truncation error.

This repository uses the undamped diagonal, which has two consequences worth
knowing when transferring hyperparameters. The multiclass working response
$r_k=-g_k/h_k$ is exactly twice XGBoost's and $\tfrac{2(K-1)}{K}$ times
LightGBM's, so a given `learning_rate` takes a correspondingly longer step.
And because the binary logistic loss of §4.2 uses $h=p(1-p)$ as the exact
scalar second derivative rather than as a truncation, `learning_rate` is a
stronger control in the multiclass path than in the binary path of this same
library. Multiclass runs should be tuned with a smaller `learning_rate`, or
guarded with `early_stopping_rounds`, rather than reusing binary settings.

That diagonal surrogate makes the quadratic separable,

$$
g^\top q+\frac12 q^\top H q
\approx\sum_{k=1}^{K}\Bigl(g_k q_k+\frac12 h_k q_k^2\Bigr),
$$

so each class is an independent weighted regression. The discarded
off-diagonal remainder of the exact quadratic is

$$
\frac12 q^\top\bigl(H-\mathrm{diag}(h)\bigr)q
=-\frac12\sum_{k\neq j}p_k p_j q_k q_j.
$$

Completing the square on each diagonal term recovers the scalar identity of
§3.1,

$$
g_k q_k+\frac12 h_k q_k^2
=\frac12 h_k\Bigl(q_k+\frac{g_k}{h_k}\Bigr)^2
-\frac{g_k^2}{2h_k},
$$

so the unrestricted diagonal Newton step is $q_k=r_k=-g_k/h_k$. As in binary
logistic loss, $\varepsilon$ is machine epsilon. With that diagonal Hessian,
the Newton working response is

$$
r_k=-\frac{g_k}{h_k}
=\frac{\mathbf{1}_{\{k=y\}}-p_k}{p_k(1-p_k)}
=
\begin{cases}
1/p_{y}, & k=y,\\
-1/(1-p_k), & k\neq y.
\end{cases}
$$

Because $0\leq p_k(1-p_k)\leq 1/4$, confidently predicted classes have small
$h_k$ and therefore small fitting weight, as in binary logistic loss.
Observation $i$ is fit for class $k$ with effective weight
$\widetilde w_{i,k}=w_i h_{i,k}$. If

$$
\widehat p_k=\frac{\sum_i w_i\mathbf{1}_{\{y_i=k\}}}{\sum_i w_i},
$$

the initial score is the clipped log prior

$$
F_{0,k}=\log\widehat p_k,
\qquad
\widehat p_k\leftarrow
\min\bigl(\max(\widehat p_k,\varepsilon),1-\varepsilon\bigr).
$$

Without clipping, $\mathrm{softmax}(\log\widehat p)=\widehat p$, so the constant
initializer matches the class prior. Clipping keeps $\log\widehat p_k$ finite
when a class is absent from a weighted subsample.

At boosting round $m$, one learner family $\mathcal H_{k_m}$ is sampled or
selected greedily. Then, independently for each class $k$,

$$
f_{m,k}\in\arg\min_{f\in\mathcal H_{k_m}}
\sum_{i=1}^{n}w_i h_{i,k}\bigl(r_{i,k}-f(x_i)\bigr)^2.
$$

Trees, RFF ridge, linear models, and neural nets all solve this same scalar
weighted problem in $(r_{\cdot,k},h_{\cdot,k})$; only the hypothesis class
changes.

A single shrinkage $\eta_m$ is applied to every class,

$$
F_{m,k}(x)=F_{m-1,k}(x)+\eta_m f_{m,k}(x).
$$

With `line_search=True`, the shared step is chosen from the same discrete grid
as in the scalar case,

$$
\eta_m\in\arg\min_{\eta\in\eta\{0.25,0.5,1,1.5,2\}}
\sum_i w_i\ell\bigl(y_i,F_{m-1}(x_i)+\eta f_m(x_i)\bigr),
$$

where $f_m(x)\in\mathbb{R}^K$ stacks the $K$ class corrections. Predicted
probabilities and labels are

$$
P(y=k\mid x)=\mathrm{softmax}(F_M(x))_k,
\qquad
\hat y(x)=\arg\max_k F_{M,k}(x)
=\arg\max_k P(y=k\mid x).
$$

The two argmaxima coincide because softmax is strictly monotone in each
coordinate relative to the others. Binary problems keep the scalar logistic
path of §4.2 so that `decision_function` remains one-dimensional. For $K=2$,
softmax on a pair of scores is equivalent to a sigmoid of their difference,

$$
\frac{e^{F_1}}{e^{F_0}+e^{F_1}}=\sigma(F_1-F_0),
$$

which is why the binary path stores a single log-odds $F$ instead of two
class scores.

Direct $e^{F_k}$ overflows for large scores. The implementation evaluates a
stable softmax by subtracting the coordinate-wise maximum
$m(x)=\max_j F_j(x)$,

$$
p_k=\frac{e^{F_k-m}}{\sum_j e^{F_j-m}},
\qquad
\ell(y,F)=m+\log\sum_j e^{F_j-m}-F_y
=\mathrm{logsumexp}(F)-F_y.
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

**$\delta$ must be chosen on the scale of the residuals.** The Newton working
response is

$$
z=-\frac{g}{h}=-e\left(1+(e/\delta)^2\right),
$$

which grows like $e^3/\delta^2$ once $\lvert e\rvert\gg\delta$. With the
default $\delta=1$ and a target whose residuals are of order $10^2$, the very
first working response is of order $10^6$, and boosting diverges rather than
converges. Standardize the target or set `objective_parameter` to roughly the
residual scale (a robust spread estimate of $y$ is a good starting point).
This mirrors the `huber_slope` parameter in XGBoost's
`reg:pseudohubererror`.

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

For multiclass targets the sampled family $K_m$ is fit once per class, so round
$m$ contributes the stack $(f_{m,K_m}^{(1)},\ldots,f_{m,K_m}^{(C)})$ described
in §4.2.1. One family is drawn per round, not one per class, so every class
receives a correction from the same inductive bias in a given round.

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

For multiclass, $f_{m,k}(x)$ is the stacked $K$-class correction from family
$k$. Candidate fitting may run in parallel. In this mode, probabilities
determine which candidates are eligible ($p_k>0$), but their magnitudes do not
enter the loss comparison.

### 5.3 Row subsampling

When `subsample=s<1`, HNBM draws without replacement

$$
n_s=\left\lceil s n_+\right\rceil
$$

eligible observations, where $n_+$ is the number having positive effective
weight $w_i h_i$. The learner is fit on that sample, while its predictions and
greedy-selection loss are evaluated on the complete training set. Multiclass
rounds subsample independently for each class $k$, using

$$
n_+^{(k)}=\bigl\lvert\{i:h_{i,k}>0\}\bigr\rvert.
$$

### 5.4 Early stopping

For validation data, HNBM records

$$
\mathcal R_{\mathrm{val}}^{(m)}
=\frac{\sum_i w_i^{\mathrm{val}}\,
\ell(y_i^{\mathrm{val}},F_m(x_i^{\mathrm{val}}))}
{\sum_i w_i^{\mathrm{val}}},
$$

where $w^{\mathrm{val}}$ comes from `eval_sample_weight` and defaults to the
uniform vector, recovering the unweighted mean.

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

1. Validate the learner pool and probabilities. Map binary labels to
   $\{-1,+1\}$, or map multiclass labels to integer codes $0,\ldots,K-1$, and
   compute $F_0$.
2. For $m=1,\ldots,M$:
   1. Compute $g_i$ and $h_i$ at $F_{m-1}(x_i)$. For multiclass these are
      $K$-vectors and the Hessian is the diagonal softmax approximation.
   2. Form $r_i=-g_i/h_i$ and $\widetilde w_i=w_i h_i$.
   3. Sample $K_m$ from the configured categorical distribution, or fit all
      eligible candidates and compare their updated losses.
   4. Optionally subsample positive-weight observations.
   5. Fit the selected cloneable regressor to $r_i$ with weights
      $\widetilde w_i$. Multiclass fits one scalar regressor per class.
   6. Use fixed shrinkage or choose $\eta_m$ from the discrete line-search grid.
   7. Add the scaled learner (or $K$ class learners) to $F_{m-1}$, record
      losses and metrics, and invoke callbacks.
3. Optionally restore the best validation ensemble.

For a multiclass round the stacked update is

$$
F_{m,k}(x)=F_{m-1,k}(x)+\eta_m f_{m,k}(x),
\qquad k=1,\ldots,K,
$$

with one scalar $f_{m,k}$ per class from the same hypothesis family.
Prediction uses exactly

$$
F_{M'}(x)=F_0+\sum_{m=1}^{M'}\eta_mf_m(x),
$$

where $M'\leq M$ after early stopping or explicit callback termination.
Multiclass $F_{M'}(x)$ and $F_0$ are $K$-vectors; binary and regression remain
scalars.

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
formula. Multiclass applies the same leaf value independently to each class
tree, using $(g_{i,k},h_{i,k})$ in place of $(g_i,h_i)$. XGBoost incorporates
the regularizer directly in leaf optimization
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
fits run in parallel. Multiclass multiplies those costs by the number of
output classes: each selected family is cloned once per class, so a round
costs about $K_{\mathrm{out}}$ times a binary round.

## 12. Implementation-specific summary

The following details describe this repository rather than every possible
HNBM implementation:

- Base learners must be cloneable regressors whose `fit` explicitly accepts
  `sample_weight`.
- Classification is binary logistic or multiclass softmax. Binary keeps a
  scalar logit; multiclass fits $K$ scalar learners per round with a diagonal
  Hessian.
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
