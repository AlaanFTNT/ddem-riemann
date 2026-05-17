# Plan B: Lindley-reframe text — SUPERSEDED, not deployed

**Status as of 2026-05-17:** SUPERSEDED. The trimmed-DDEM Bayes factor came in at ln B = +1.45 ± 0.38 (inconclusive band [-2, +2] with mild positive lean). No Lindley reframe required. §7.2 of the paper now reports both ln B values (full model -4.41, canonical reduction +1.45) under the standard model-comparison framing. This file is retained as a record of the contingency plan that was prepared but not used.

---

(Original Plan B prose below — do not deploy.)

---

## §7.2 — replacement opening paragraph

The joint posterior on the 7-parameter DDEM places the dimensionless coupling at β₀ = X.X × 10⁻³ ± X.X × 10⁻³ (68% CI), corresponding to a frequentist statement that β₀ exceeds zero at ~Nσ. The canonical-reduction posterior at α = 1/2 and w_DE = −1 fixed (§3.5) gives β₀ = X.X × 10⁻³ ± X.X × 10⁻³ under the same data. The Bayesian model comparison, however, returns ln B(DDEM/ΛCDM) = −4.41 ± 0.41 for the 7-parameter model and ln B(trimmed/ΛCDM) = X.X ± X.X for the canonical reduction. The two metrics carry different information: the frequentist statement registers a residual in the data that prefers nonzero coupling; the Bayesian statement registers that the additional prior volume integrates over more low-likelihood regions than it adds high-likelihood ones. Both are correct interpretations of the present data.

This is a Lindley-paradox-like split, in the Berger-Sellke (1987) sense: a model parameter that frequentist hypothesis-testing rejects at the canonical threshold can simultaneously be disfavored by Bayesian model comparison once the prior volume is integrated over. Resolution requires either tighter priors (constraining ε, α, w_DE on physical grounds, as in §3.5) or higher-precision data that narrows the posterior enough to drive the Bayesian and frequentist conclusions back into agreement.

## §8 — replacement closing paragraph

The framework's falsifiability program does not depend on the current Bayes factor. The three-tier hierarchy (§4.3) tests the *structure* of the proposed spectrum — its GUE pair-correlation statistics, the alignment of P(k) residual peaks with k_n = γ_n / η₀, and the low-n Riemann-specific signature distinguishing γ_n from rival GUE-class spectra. Each tier produces a binary verdict independent of whether current data prefers ΛCDM in a Bayesian model comparison. DESI DR3 brings the high-n comb into Tier 2 sensitivity range; Euclid DR1 adds independent lensing constraints on the same wavenumbers; SKA-Low 21 cm intensity mapping is the route to Tier 3. The current generation of data is not the test the framework was designed for. The paper's contribution is the falsifiable construction, the demonstration that current data already excludes the most parameter-extended versions of the model, and the calibrated forecast for the surveys that can deliver the actual test.

## §9 — replacement conclusion sentence

The current joint fit places the coupling β₀ at ~Nσ above zero in frequentist terms and disfavors the 7-parameter model by ln B = −4.4 in Bayesian terms; the canonical-reduction model lands at ln B = X.X. The framework's three-tier falsifiability program is unaffected by these tensions and is calibrated against DESI DR3, Euclid DR1, and SKA-Low.

---

## Notes on tone

- "Lindley paradox" appears once, as an explicit citation to Berger-Sellke 1987. No further hedging.
- Avoid: "remarkably," "perhaps surprisingly," "of course." Just state the dual result and move on.
- Don't include the Bayes factor in the abstract bottom line unless the trimmed result still disfavors at substantial level. The abstract should emphasize the falsifiability program, not the current evidence verdict.

## Replacement-window indicators

- §7.2 paragraph: replace lines ~248–260 of Paper_1_Draft.md (β₀ posterior table + interpretation).
- §8.x: insert a new §8.4 if Lindley path; replace §8.1 closing if shorter version.
- §9 conclusion: replace the headline sentence about β₀ detection.
