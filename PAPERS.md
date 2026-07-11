# Annotated bibliography

This is a guided tour of the literature behind the thesis *From Rough to Lifted*.
Each entry says what the paper is and, concretely, how the thesis relies on it.
References are grouped by theme and tagged by how load-bearing they are:

- **(central)** the argument or the code depends on it directly;
- **(supporting)** used for a specific derivation, method, or framing;
- **(peripheral)** background or related work, cited once or listed only.

For the exact equations and numerical schemes, see
[`docs/MODEL_SPEC.md`](docs/MODEL_SPEC.md); for the full reference list, see the
bibliography of [the thesis](docs/From_Rough_to_Lifted.pdf).

---

## The two models compared

### Abi Jaber (2019), Lifting the Heston model **(central)**
*Abi Jaber, E. (2019). Lifting the Heston model. Quantitative Finance 19(12), 1995-2013. arXiv:1810.04868.*

Introduces the lifted Heston model, a finite-dimensional Markovian approximation
of rough Heston built as a weighted sum of n mean-reverting CIR-type factors whose
kernel approximates the fractional kernel via a Laplace representation on a
geometric grid. It converges to rough Heston as n grows while staying tractable
through n coupled Riccati ODEs and an affine characteristic function. This is one
of the two core models the thesis compares and the one it extends, so it is
load-bearing throughout: the thesis implements the model from scratch (Riccati
solver, affine characteristic function, COS Fourier inversion), reuses its
geometric grid and the practical n=20, r_n=2.5 setting, invokes its existence and
convergence theorems, and reproduces its convergence and ATM-skew claims,
including a dedicated appendix replicating Figure 1 on the paper's own reference
date 2018-06-20. The proposed two-factor extension is built on this lifted
construction to inherit Riccati-Fourier pricing.

### Zhu, Loeper, Chen and Langrené (2021), Markovian approximation of rough Bergomi **(central)**
*Zhu, Z., Loeper, G., Chen, W., and Langrené, N. (2021). Markovian approximation of the rough Bergomi model for Monte Carlo option pricing. Mathematics 9(5), 528.*

Proposes a Markovian approximation of the rough Bergomi model by replacing the
fractional power-law kernel with a sum of exponential kernels whose weights and
mean-reversion speeds come from an L2 kernel-fitting objective, giving a
finite-dimensional Markovian state for Monte Carlo option pricing of the lognormal
variance process. This is the second pillar of the thesis: the SPX comparison pits
the lifted Heston against this affine Bergomi (aBergomi) model. The thesis
implements aBergomi with its L2-optimised kernel, derives its ATM skew via the
Bergomi-Guyon formula, tabulates its structural contrasts with lifted Heston
(lognormal vs square-root variance, Monte Carlo vs closed-form Riccati-Fourier
pricing), and uses its kernel construction as the counterpoint in both the
paper-faithful and kernel-controlled comparison layers.

## Rough volatility: foundations

### Gatheral, Jaisson and Rosenbaum (2018), Volatility is rough **(supporting)**
*Gatheral, J., Jaisson, T., and Rosenbaum, M. (2018). Volatility is rough. Quantitative Finance 18(6), 933-949.*

The paper that launched rough volatility: it shows empirically that log realised
variance behaves like a fractional Brownian motion with Hurst index well below
1/2 (near 0.1), so variance is driven by a fractional process and the models are
non-Markovian. The thesis cites it for framing rather than derivation: it is
credited with introducing rough volatility models and their key advantage of
reproducing the short-maturity ATM-skew explosion, and it grounds the empirical
stylised fact that the skew decays as a power law. The thesis does not use its
estimation method directly.

### El Euch and Rosenbaum (2019), The characteristic function of rough Heston **(supporting)**
*El Euch, O., and Rosenbaum, M. (2019). The characteristic function of rough Heston models. Mathematical Finance 29(1), 3-38.*

Introduces the rough Heston model, replacing the Brownian-driven CIR variance with
fractional Volterra dynamics of Hurst index H in (0, 1/2) and singular kernel
(t-s)^(H-1/2), and derives its characteristic function as the solution of a
fractional Riccati equation, giving a semi-closed Fourier pricing route despite
the loss of the Markov and semimartingale properties. The thesis relies on it as
the defining source of the rough Heston model: it writes down the variance SDE
that lifted Heston is built to approximate, and implements the fractional Riccati
plus Fourier inversion as the exact "rough" ground truth
([`src/rough_heston`](src/rough_heston)) against which the Markovian lift is
measured.

### Bayer, Friz and Gatheral (2016), Pricing under rough volatility **(supporting)**
*Bayer, C., Friz, P. K., and Gatheral, J. (2016). Pricing under rough volatility. Quantitative Finance 16(6), 887-904.*

Introduces the rough Bergomi model, a lognormal forward-variance model whose
variance is driven by a Riemann-Liouville fractional Brownian motion, and
documents that the SPX ATM skew behaves as a power law with exponent in (0, 1/2).
The thesis cites it as empirical grounding for the short-maturity skew explosion
and as the rough model that the aBergomi model approximates, providing the
reference dynamics and first-order skew formula the approximation must reproduce.

## The roughness debate

### Abi Jaber and Li (2025), Volatility models in practice **(central)**
*Abi Jaber, E., and Li, X. (2025). Volatility models in practice: Rough, path-dependent or Markovian? arXiv:2401.03345.*

A large-scale empirical study calibrating rough, path-dependent and Markovian
models on more than 2,800 daily SPX surfaces (2011 to 2022). It finds that rough
Bergomi models are constrained by the universal single-kernel skew scaling
psi(T) proportional to T^(H-1/2) and are therefore too rigid to fit the short and
long ends of the skew term structure simultaneously, while simple two-factor
Markovian Bergomi models outperform them on the global surface. This is the
thesis's primary empirical motivation: the single-kernel rigidity diagnosis is the
explicit target that the proposed two-factor lifted Heston is designed to break.
The thesis also adopts the paper's data preprocessing (put-call-parity forwards,
SSVI interpolation, no-arbitrage filtering, forward-variance bootstrapping) and
adapts its daily-panel protocol into a smaller temporal out-of-sample study, while
filling a gap it identifies: the paper compares rough versus Markovian only within
the Bergomi family, never against lifted Heston.

### Cont and Das (2023), Rough volatility: fact or artefact? **(central)**
*Cont, R., and Das, P. (2023). Rough volatility: fact or artefact? arXiv:2203.13820.*

Argues that the empirical evidence for rough volatility (estimated Hurst index near
0.1) may be a statistical artefact: spot volatility is unobserved, so
realised-variance estimators from high-frequency returns carry a finite-sample and
microstructure error that biases roughness estimators downward, making realised
volatility appear rough even when the latent spot volatility is a Brownian
semimartingale. The thesis engages this substantively: it devotes a full
subsection to reconstructing the realised-variance-plus-error decomposition and the
p-variation critical-exponent argument, read alongside Abi Jaber's complementary
lifted-Heston experiment. It uses the critique to stress that its calibrated H is
a model selector under the risk-neutral measure Q, not a measurement of
physical-measure regularity, and positions its own H-identifiability finding as
logically independent of and complementary to the Cont and Das P-measure critique.

### Abi Jaber, Illand and Li (2022), The quintic Ornstein-Uhlenbeck model **(peripheral)**
*Abi Jaber, E., Illand, C., and Li, S.-X. (2022). The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX and VIX smiles. Risk Magazine.*

Introduces the quintic Ornstein-Uhlenbeck volatility model, a continuous
Markovian (non-rough, non-affine) model in which volatility is a polynomial of a
single OU factor, and shows it jointly calibrates SPX and VIX smiles, a benchmark
in the debate over whether roughness is necessary. In this thesis the reference is
essentially background: it is listed as related literature on alternative
Markovian models but supports no model, method, or argument in the text (the
thesis calibrates only SPX surfaces).

## Markovian lifts and affine theory

### Abi Jaber, Larsson and Pulido (2019), Affine Volterra processes **(supporting)**
*Abi Jaber, E., Larsson, M., and Pulido, S. (2019). Affine Volterra processes. Annals of Applied Probability 29(5), 3155-3200.*

Establishes the general theory of affine Volterra processes: stochastic Volterra
equations with completely monotone kernels admit a Fourier-Laplace transform
governed by a Riccati-Volterra equation, with existence, uniqueness and
non-negativity under an admissibility condition on the kernel. The thesis leans on
it as the tractability foundation for its Markovian models: it certifies that the
lifted Heston Riccati block (a finite positive sum of exponentials, hence
completely monotone) and its variants (the L2-optimised kernel, a bimodal
two-Hurst kernel) remain well-posed and positivity-preserving. It also marks the
scoping boundary: the proposed two-factor extension is flagged as falling outside
this scalar affine-Volterra framework, which motivates a separate positivity
analysis.

### Cuchiero and Teichmann (2018), Markovian lifts of stochastic Volterra processes **(peripheral)**
*Cuchiero, C., and Teichmann, J. (2018). Generalized Feller processes and Markovian lifts of stochastic Volterra processes: the affine case. arXiv:1804.10450.*

Uses the generalized Feller process framework to construct Markovian lifts of
stochastic Volterra processes in the affine case, giving a rigorous foundation for
representing rough affine dynamics as infinite-dimensional Markov processes. The
thesis cites it once, in the outlook chapter, as the framework in which a proposed
lifted Wishart (Volterra Wishart) model would live: a matrix Wishart SDE whose
integrals are replaced by Volterra convolutions against a rough kernel. This
direction is explicitly flagged as beyond the scope of the thesis.

### Cuchiero, Filipović, Mayerhofer and Teichmann (2011), Affine processes on positive semidefinite matrices **(peripheral)**
*Cuchiero, C., Filipović, D., Mayerhofer, E., and Teichmann, J. (2011). Affine processes on positive semidefinite matrices. Annals of Applied Probability 21(2), 397-463.*

Develops the theory of affine processes valued in the cone of symmetric positive
semidefinite matrices, characterising their generators and the matrix Riccati
equations that make their transform exponentially affine in the matrix state. The
thesis cites it once, in the future-work outlook, to certify that a proposed
Wishart variance SDE is affine in this precise sense. This underwrites the argument
that lifting the scalar two-factor variance to a matrix state dissolves the
non-affine sqrt(V1 V2) cross term that breaks the affine ansatz in the correlated
scalar model. It supports theoretical framing of a proposed extension only.

### Da Fonseca, Grasselli and Tebaldi (2008), A multifactor volatility Heston model **(peripheral)**
*Da Fonseca, J., Grasselli, M., and Tebaldi, C. (2008). A multifactor volatility Heston model. Quantitative Finance 8(6), 591-604.*

Develops a multifactor Heston model in which the scalar variance is replaced by a
matrix-valued Wishart process on the cone of positive semidefinite matrices, giving
a tractable affine model with a matrix Riccati transform. The thesis cites it once,
in the outlook's Wishart subsection, as the principled resolution to a fragility it
identifies in its own scalar two-factor extension: correlated variance drivers
produce a non-affine cross term, and lifting to a matrix state carries the
correlation as a linear coordinate and restores affineness. Used to frame future
work, not part of the models actually calibrated.

### Cuchiero, Keller-Ressel and Teichmann (2012), Polynomial processes **(peripheral)**
*Cuchiero, C., Keller-Ressel, M., and Teichmann, J. (2012). Polynomial processes and their applications to mathematical finance. Finance and Stochastics 16(4), 711-740.*

Introduces polynomial processes, a class of Markov processes generalizing affine
processes for which conditional moments of polynomial payoffs stay polynomial in
the state and are computable by finite-dimensional linear algebra. It sits in the
same affine/polynomial-process family that grounds the lifted Heston model's
tractability, but it is background context: it appears in the bibliography without
a citation site in the body.

## Numerical methods

### Fang and Oosterlee (2009), the COS method **(supporting)**
*Fang, F., and Oosterlee, C. W. (2009). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing 31(2), 826-848.*

Introduces the COS method, which prices European options by expanding the truncated
log-return density in a Fourier-cosine series whose coefficients are read directly
off the characteristic function, giving fast, spectrally accurate prices with
analytic payoff coefficients. The thesis adopts it as the pricing engine for all
affine (lifted and rough) Heston computations
([`src/common/cos_method.py`](src/common/cos_method.py)): once the Riccati system
yields the characteristic function, prices at every strike come from the COS sum,
then invert to implied vols via Brent. The thesis follows the paper's practical
guidance, using N_cos = 256 terms and a truncation multiplier L0 = 12 to absorb
the heavier left tail at small H.

### Gatheral and Jacquier (2014), Arbitrage-free SVI surfaces **(supporting)**
*Gatheral, J., and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative Finance 14(1), 59-71.*

Introduces arbitrage-free parametrisations of the implied volatility surface: the
raw SVI slice and the surface-level SSVI, with conditions ruling out butterfly and
calendar arbitrage. The thesis uses it as tooling in two ways. First, a per-slice
raw SVI fit ([`src/data/svi.py`](src/data/svi.py)) supplies the smooth analytic
ATM-skew estimator behind the skew term-structure figures, complementing the
98/102 finite-difference skew used in the tables. Second, SSVI interpolation fills
sparse strike grids during SPX surface preprocessing. It is a curve-fitting tool,
not one of the models compared.

### Lord and Kahl (2007), Optimal Fourier inversion **(peripheral)**
*Lord, R., and Kahl, C. (2007). Optimal Fourier inversion in semi-analytical option pricing. Journal of Computational Finance 10(4), 1-30.*

Studies Fourier-based semi-analytical option pricing and how to choose the optimal
damping parameter so inversion of the characteristic function stays numerically
stable across strikes and maturities. This is directly relevant in spirit, since
the thesis prices affine models by Fourier inversion, but in practice the thesis
uses the COS method of Fang and Oosterlee (2009) for that inversion. Lord and Kahl
appears in the bibliography as background reading, not a method the thesis builds
on.

## Classical models and economic background

### Heston (1993), the classical stochastic volatility model **(supporting)**
*Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. Review of Financial Studies 6(2), 327-343.*

Introduces the classical one-factor stochastic volatility model with square-root
(CIR) variance and its closed-form option pricing via a scalar Riccati ODE for the
characteristic function. The thesis uses it in three ways: as the motivating
benchmark whose flat short-maturity ATM skew fails to match the observed power-law
explosion (which justifies rough volatility); as the canonical Markovian template
that the lifted approximation generalises; and as the closed-form scalar ansatz
reused for the second block of the proposed two-factor extension, where an ordinary
Heston Riccati is solved in parallel with the lifted block.

### Bergomi and Guyon (2012), the ATM skew expansion **(central)**
*Bergomi, L., and Guyon, J. (2012). Stochastic volatility's orderly smiles. Risk 25(5), 60-66.*

A first-order-in-vol-of-vol expansion that approximates the smile, and in
particular the ATM skew, via the instantaneous covariation between log-spot and the
forward-variance curve. The thesis elevates it to a named theorem whose Heston-type
corollary rewrites the ATM skew as a twice-integrated kernel, making kernel shape
the sole driver of short-maturity skew. This formula is the analytical backbone of
the comparison: it diagnoses the bounded short-maturity skew of classical Heston
and one-factor Bergomi, the T^(H-1/2) blow-up of rough and lifted kernels, and the
ATM skew of the proposed two-factor extension.

### Christoffersen, Heston and Jacobs (2009), why multifactor SV works **(supporting)**
*Christoffersen, P., Heston, S., and Jacobs, K. (2009). The shape and term structure of the index option smirk: Why multifactor stochastic volatility models work so well. Management Science 55(12), 1914-1932.*

Shows empirically that multifactor square-root stochastic volatility, with total
variance written as the sum of two independent CIR processes, captures the shape
and term structure of the SPX smirk far better than one-factor Heston. The thesis
cites it as one of the two structural inspirations for its two-factor extension
(alongside Abi Jaber's lifted Heston): the extension adopts this paper's
double-Heston price form, with the return variance a sum of two square-root
diffusions each paired with its own independent block, which is what preserves the
affine structure needed for transform pricing.

### Bergomi (2016), Stochastic Volatility Modeling **(peripheral)**
*Bergomi, L. (2016). Stochastic Volatility Modeling. Chapman and Hall/CRC Financial Mathematics Series.*

The textbook that develops the forward-variance approach to stochastic volatility,
prescribing dynamics directly on the forward-variance curve rather than on spot
variance, with the one-factor and n-factor lognormal Bergomi models as its central
objects. The thesis cites it once, to introduce the classical Bergomi model whose
finite short-maturity skew motivates replacing the exponential kernel by a rough
power-law kernel. Background rather than load-bearing.

### Andersen and Piterbarg (2007), Moment explosions **(peripheral)**
*Andersen, L. B. G., and Piterbarg, V. V. (2007). Moment explosions in stochastic volatility models. Finance and Stochastics 11(1), 29-50.*

Derives conditions under which moments of the underlying explode in finite time in
stochastic volatility models such as Heston, with consequences for option pricing
and implied-volatility asymptotics. In this thesis it is peripheral: it appears as
a bibliography entry and reflects moment-explosion theory as background for the
affine models studied, but no specific result relies on it.

### Black (1976) and Christie (1982), the leverage effect **(peripheral)**
*Black, F. (1976). Studies of stock price volatility changes. Proceedings of the 1976 Meeting of the Business and Economic Statistics Section, American Statistical Association, 177-181.*
*Christie, A. A. (1982). The stochastic behavior of common stock variances: Value, leverage and interest rate effects. Journal of Financial Economics 10(4), 407-432.*

Two classic empirical studies documenting that equity return volatility rises as
prices fall, the observation behind the "leverage effect" interpretation. The
thesis cites them together, once, in its economic-motivation discussion of the
equity skew, to attribute the negative return-volatility correlation to leverage.
They are presented as one of two classical readings of the asymmetry, alongside
Rubinstein's crashophobia, with the thesis explicitly declining to adjudicate.

### Rubinstein (1994), crashophobia **(peripheral)**
*Rubinstein, M. (1994). Implied binomial trees. Journal of Finance 49(3), 771-818.*

Introduces implied binomial trees, fitting a risk-neutral terminal distribution to
observed option prices, and documents the persistent negatively-skewed index smile
that emerged after the 1987 crash, attributing it to a standing premium against
sudden large declines ("crashophobia"). The thesis cites it once, as one of two
classical readings of the return-volatility asymmetry alongside the leverage
effect. Background context only.
