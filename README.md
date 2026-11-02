# C3: Committed Confidence Calibration

Selection-conditional calibration audit, parallel-commitment error decomposition, and calibrated decoding for masked diffusion language models (DLMs).

- **Paper:** [`paper/main.pdf`](paper/main.pdf) (built from the self-contained [`paper/main.tex`](paper/main.tex) via `cd paper && make -B` using `tectonic`).
- **Presentation slides:** [`slides/slides.pdf`](slides/slides.pdf) (built from the self-contained [`slides/slides.tex`](slides/slides.tex) via `cd slides && make -B` using `tectonic`).
- **Pre-registration and deviations:** [`prereg/frozen_config.json`](prereg/frozen_config.json), [`prereg/analysis_commands.sh`](prereg/analysis_commands.sh), [`prereg/deviations.md`](prereg/deviations.md).
- **Cluster execution notes:** [`slurm/CLUSTER.md`](slurm/CLUSTER.md).

## Problem and Contributions

Masked diffusion LMs decode by iteratively unmasking the highest-confidence positions (for example, top-$k$ confidence selection or threshold rules such as Fast-dLLM). Standard calibration checks evaluate random masks without conditioning on the selection step. Because per-position confidence estimates contain noise, taking a maximum over $m$ masked candidates can introduce a winner's curse, inflating committed-token confidence relative to accuracy.

We define the committed-token gap as $\Delta(m) = \mathbb{E}_m[c - a]$ (confidence minus accuracy, positive = overconfident). Under conditional independence of position errors given state $s$, Proposition 2 splits the gap into own miscalibration plus a context-dependent field term:

$$\Delta(m) = \mathbb{E}_m[\sigma(\ell) - \hat{\pi}(\ell)] + \mathbb{E}_m[\hat{\pi}(\ell) - \hat{\pi}(\ell, s)] + \mathbb{E}_m[\hat{\pi}(\ell, s) - a]$$

where $\ell = \log(c / (1 - c))$ is the token confidence log-odds and the third term captures within-state shared errors.

### C1: Selection-Conditional Calibration Audit

- **Models and precision:** LLaDA-8B (Base and Instruct, fp16), Dream-7B (Base and Instruct, fp16), and a 110M-parameter bidirectional MDLM (`c3/training/model.py`: 12 layers, width 768, 12 heads, sequence length 256, GPT-2 BPE + `[MASK]`, trained for 169,055 steps / 10.0B tokens on FineWeb-Edu sample-10BT in fp32, achieving a held-out NELBO perplexity bound of 37.2 [36.5, 38.0] vs. GPT-2 small 28.5 [28.0, 29.0] across 2,000 held-out documents).
- **Datasets and splits:** 2,000 sampled FineWeb-Edu documents (Base and 110M; 1,979 for LLaDA/110M and 1,976 for Dream after the 256-token length filter), GSM8K test set (1,319 problems: 660 fit / 659 report), MBPP test set (499 problems up to 1,024 tokens: 249 fit / 250 report), a 500-document post-release English Wikipedia time-split set (2025-04-04 to 2026-08-29), and a 198-document 4-bit (NF4) precision check on LLaDA-8B-Base. Across all 11 trace runs: 0.49B position rows and 208 GPU-hours.
- **Endpoints and inference:** Teacher-forced traces under uniform masks ($t \in \{0.1, \dots, 0.9\}$), confidence top-$k$ decoder states ($k \in \{2, 4, 8\}$), and matched uniform masks. Random pools of size $m \in \{1, 2, 4, 8, 16, 32\}$ (20 pools per state, random tie-breaks) yield primary contrasts $\text{E1} = \Delta(32) - \Delta(1)$ and $\text{E2} = \text{field}(32) - \text{field}(1)$, using cross-fitted isotonic $\hat{\pi}(\ell)$ and 9-knot spline-logistic $\hat{\pi}(\ell, s)$ calibrators, 10,000 document-bootstrap resamples, and Holm correction ($\alpha = 0.05$, 14 tests per family).
- **Verified results:**
  - In $k=4$ decoder states, E1 is positive and claimed (declared and Holm-rejected) in all 7 model-task arms, ranging from $+0.006$ [$0.005, 0.008$] (LLaDA-8B-Base) and $+0.009$ [$0.007, 0.011$] (MDLM-110M) to $+0.156$ [$0.126, 0.189$] (LLaDA-8B-Instruct, MBPP) and $+0.186$ [$0.151, 0.224$] (Dream-7B-Instruct, MBPP).
  - In uniform states, E1 is negative for both base models ($-0.003$, $-0.002$) and LLaDA-8B-Instruct on GSM8K ($-0.022$), and positive for Dream-7B-Instruct on GSM8K ($+0.061$) and MBPP ($+0.257$ [$0.227, 0.290$]).
  - E2 is positive in 7 of 8 instruct cells ($0.013$ to $0.176$, shrinking to $0.000$ to $0.098$ under an additive calibrator without $\ell \times \text{feature}$ interactions), while MBPP cells show a large within-state residual term ($0.054$ to $0.100$). Base and 110M models stay below $0.005$ in $|\text{E2}|$, near the synthetic null bias.
  - On tokens committed by the $k=4$ decoder with 32 to 63 masked positions remaining, the descriptive native gap ($c - a$) is 1.8 to 4.4 times each arm's random-mask 15-bin ECE.

### C2: Dependency vs. Estimation Error in Parallel Commitment

- **Decomposition:** Committing a subset $S$ in parallel incurs $\mathrm{KL}(p^* \,\|\, \prod_{i \in S} p_{\theta,i}) = \mathrm{TC}(x_S \mid x_{\text{obs}}) + \sum_{i \in S} \mathrm{KL}(p^*_i \,\|\, p_{\theta,i})$, separating (A) dependency error (total correlation) from (B) marginal estimation error.
- **Synthetic JSON probe:** 200 SchemaStore schemas (6 to 16 single-token slots per schema, pairwise coupling probability $\rho \in \{0, 0.25, 0.5, 0.75, 1\}$, 1,000 evaluation instances) evaluated with the fine-tuned 110M MDLM (`ft110m-v3-visible`, 300M fine-tuning tokens on 200k probe instances, validation NELBO $0.1751$).
  - Dependency (A) overtakes estimation (B) only at $\rho \ge 0.5$ and multi-token steps ($k=8$ for $\rho \in \{0.5, 0.75\}$; $k=4$ at $\rho=1$ under confidence top-$k$). At $\rho=1, k=8$ (confidence top-$k$), (A) reaches $0.338$ [$0.320, 0.355$] nats/slot vs. (B) $0.109$ [$0.095, 0.123$] (76% of total KL). For the untuned 110M model, (B) dominates everywhere ((A) $\le 4\%$).
- **Real-text proxy:** Parallel-sequential top-token disagreement on 300 fit-half FineWeb-Edu documents across mask rates $0.1$ to $0.9$ is $0.0021$ (110M) and $0.0022$ (LLaDA-8B-Base) at $k=8$, and at most $0.0003$ for $k \le 4$. On the probe, Spearman correlation between the disagreement proxy and exact (A) is $0.49$ [$0.28, 0.63$].

### C3: Calibrated Commitment Policy (C3a)

- **Policy:** At each decoding step, C3a computes $\hat{c}_i = \hat{\pi}(\ell_i, s)$ from a fit-half `ContextLogistic` calibrator (using median $\ell$, mask rate, and $\log|M|$ over the generated region) and commits the top $n = \max(1, |\{i : \hat{c}_i \ge 1 - \alpha\}|)$ masked positions in the active 32-token block ordered by $\ell_i$, requiring no extra forward pass.
- **Calibration improvement:** On held-out report-half teacher-forced decoder states ($k=4$), equal-mass ECE drops from $0.060$ to $0.004$ on GSM8K and from $0.040$ to $0.005$ on MBPP.
- **Matched-speed free generation (exploratory):** Evaluated on LLaDA-8B-Instruct (fp16, 256 tokens, block size 32, greedy) on report-half GSM8K ($N=659$) and MBPP ($N=121$ completed within compute budget; an $N=250$ sensitivity run is included in [`artifacts/t12/sensitivity/mbpp250/`](artifacts/t12/sensitivity/mbpp250/)).
  - **GSM8K at speed 2 (~4 tok/step):** C3a ($0.789$ accuracy at $4.04$ tok/step, $\alpha=0.271$) improves over fixed-count baselines Top-$k$ ($0.730$, $+0.059$ [$0.026, 0.093$]), Margin ($0.736$, $+0.053$ [$0.020, 0.088$]), and PC-Sampler ($0.707$, $+0.082$ [$0.050, 0.112$]), and matches raw Threshold ($0.791$, $-0.002$ [$-0.014, 0.012$]) and Entropy-bounded ($0.794$, $-0.005$ [$-0.035, 0.026$]). At speed 3 (~8 tok/step), C3a trails raw Threshold ($-0.064$ [$-0.106, -0.027$]).
  - **MBPP ($N=121$) at speed 2:** C3a ($0.355$ at $3.61$ tok/step, $\alpha=0.322$) matches Threshold ($0.355$, $3.73$ tok/step) and Entropy-bounded ($0.355$, $3.83$ tok/step).

Committed manifests, validation reports, probe grids, and experiment summaries are catalogued in [`artifacts/README.md`](artifacts/README.md).

## Repository Structure

```text
c3/
  __init__.py         Pre-registered design constants (POOL_SIZES, N_POOLS, N_BOOT, etc.)
  order_stats.py      Expected normal order statistics z_{r,m} for Eq. 2
  splines.py          Restricted cubic splines for confidence curve f(ell)
  design.py           Post-hoc pool draws and rank assignment with random tie-breaking
  pools.py            Pool sampling helpers and document-level aggregations
  logit.py            Logistic IRLS with ridge-penalized high-cardinality state fixed effects
  frame.py            Position-level and pool-row frames, deterministic fit/report splits
  regression.py       Eqs. 2-3 with/without state FE, cross-fitting, document bootstrap
  stats.py            Document-first means, shared bootstrap matrices, Holm correction
  metrics.py          Equal-mass ECE, smooth ECE, reliability bins, PAV isotonic regression
  calibrators.py      IsotonicOwn and ContextLogistic calibrators with pool weights
  controls.py         T8 covariates (position, token type, per-bin temperature)
  gates.py            Pre-registered G1/G2 progression gates
  disagreement.py     T10 parallel-sequential disagreement and (B) proxies
  assemble.py         Stored parquet/dataset traces to analysis frames
  provenance.py       Git commit, package versions, host, GPU, and SLURM metadata stamps
  endpoints/          E1 (Delta(m)), E2 (field-term split), and E3 (trajectory shift)
  probe/              T9 synthetic SchemaStore JSON probe, exact (A)/(B) KL, and grid runner
  decode/             T12 generation loop and decoders (TopK, Margin, Threshold, Entropy, KLASS, PC-Sampler, C3a)
  traces/             Mask generators, model adapters (LLaDA, Dream, 110M), schema-v2 collector
  data/               FineWeb-Edu streaming, GSM8K/MBPP loaders, contamination, time-split Wikipedia
  training/           110M MDLM architecture, SUBS loss, resumable trainer, compute budget
  sim/                Synthetic DGPs with known truth for estimator validation
scripts/              CLI entry points for tracing, training, probes, scoring, and paper outputs
slurm/                SLURM scripts, environment build, and cluster runbook
prereg/               Frozen config, analysis commands, T12 knob grids, and dated deviations
paper/                Self-contained ACL-style LaTeX source (main.tex), BibTeX, and main.pdf
slides/               Self-contained Beamer presentation source (slides.tex) and slides.pdf
artifacts/            Committed manifests, validation outputs, probe grids, C1/C2/C3 summaries
notebooks/            Exploratory Dorigo winner's-curse verification notebook
tests/                Pytest suite (unit, pipeline, shell/SLURM, and paper-output tests)
```

## Setup and Usage

### Environment and Tests

Requires Python >= 3.11 (tested with 3.12). Install the analysis and test dependencies without GPU packages:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,analysis]"
.venv/bin/python -m pytest tests/ -q
```

Tests that depend on `torch` skip automatically when `torch` is not installed. For trace collection, 110M training, or T12 decoding on GPUs, include the `models` extra (`transformers==4.46.3` is pinned for LLaDA/Dream `trust_remote_code` compatibility):

```bash
uv pip install --python .venv/bin/python -e ".[models,dev,analysis]"
```

### Key Commands

```bash
# Validate Eq. 2/3 and E1/E2 estimators on synthetic traces with known truth
.venv/bin/python scripts/validate_estimator.py --out artifacts/estimator_validation.json
.venv/bin/python scripts/analyze_traces.py --traces sim:context --split half --out /tmp/sim

# Check 110M MDLM compute budget (no GPU or torch required)
.venv/bin/python scripts/train_110m.py --budget-only

# Generate tables, figures, and LaTeX macros from experiment outputs
.venv/bin/python scripts/figures_t9.py
.venv/bin/python scripts/macros_t10.py
.venv/bin/python scripts/c1_paper_outputs.py --results <c1_results_dir>
.venv/bin/python scripts/c3_paper_outputs.py --records <t12_records> --verdicts <verdicts.jsonl>
```

### Reproducing Paper Outputs

| Paper section | Generator script | Source inputs |
|---|---|---|
| Section 4 (C1 tables, figures, macros) | `scripts/c1_paper_outputs.py` | Per-arm `results.json` from C1 sweep |
| Section 5 (C2 probe and real-text proxy) | `scripts/figures_t9.py`, `scripts/macros_t10.py` | `artifacts/probe/`, `artifacts/c2/` |
| Section 6 (C3 calibrator and T12 matched speed) | `scripts/c3_paper_outputs.py` | T12 sweep records, `scripts/score_t12.py` |
| Section 6 (realised risk and held-out ECE) | `scripts/c3_realised_risk.py` | Report-half teacher-forced decoder states |
| MBPP $N=250$ sensitivity check | `scripts/c3_paper_outputs.py --allow-mixed-machines` | `artifacts/t12/sensitivity/mbpp250/` |
