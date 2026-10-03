# Figure / table generation spec (for paper/scripts/make_figures.py and make_tables.py)

All inputs are small CSV/JSON files under `paper/data/`. Scripts must read them and never hard-code any
numeric result. Run with `python3` (pandas, numpy, matplotlib available locally). Outputs:
figures → `paper/figures/*.pdf`, tables → `paper/tables/*.tex` (booktabs; a bare `tabular`/`tabular*`
environment only — no `table` float, no caption; the paper wraps them).

## Display-name maps (use everywhere)

Methods (`method`, `a`, `b` columns):
| raw | display | role |
|---|---|---|
| baro | BARO | published |
| circa | CIRCA | published |
| e_diagnosis, epsilon_diagnosis | $\epsilon$-Diagnosis | published (same method; two raw names in different families — merge into one row) |
| rcd | RCD | published |
| causalrca | CausalRCA | published |
| counterfactual_attribution | Counterfactual | published |
| simplerca_author | SimpleRCA | published |
| tracerca | TraceRCA | published |
| microrank | MicroRank | published |
| mmbaro | BARO (multi-source) | published |
| max_z | max-$|Z|$ | probe |
| alertcount | Alert-count | probe |
| cd1min | CD-1min | probe |
| baseline | Traversal | shipped baseline |
| ranked_correlation | Ranked correlation | shipped baseline |
| random_walk | Random walk | shipped baseline |

Subsystems (`sub`, `subsystem_id`, `subsystem` columns):
RE1::OB→RE1-OB, RE1::SS→RE1-SS, RE1::TT→RE1-TT, RE2::OB→RE2-OB, RE2::SS→RE2-SS, RE2::TT→RE2-TT,
openrca::bank→Bank, openrca::mkt1→Market-1, openrca::mkt2→Market-2, openrca::tel→Telecom,
petshop::high_traffic→High, petshop::low_traffic→Low, petshop::temporal_traffic1→Temporal-1,
petshop::temporal_traffic2→Temporal-2.

Families: RE1→"RCAEval RE1", RE2→"RCAEval RE2", openrca→"OpenRCA", petshop→"PetShop".

Layers: `L1` = published vs published; `L2` = all pairs incl. probes and shipped baselines.
Always filter `variant == "mean"` when the column exists (seed-averaged version).

Number format: 3 decimals (0.848); signed deltas with explicit sign (+0.304 / −0.522, use `$-$` in LaTeX);
CIs as `[lo, hi]` with 3 decimals.

## Style
matplotlib, `font.family=serif`, font size 8–9 pt, colour-blind-safe palette (Okabe–Ito), vector PDF,
`bbox_inches="tight"`. Figure width: full = 5.5 in (acmsmall text width), half = 2.7 in.

## Figure 1 — `figures/fig1_forest.pdf` (full width, ~4 in tall)
Three vertically stacked or side-by-side panels: RE1 (10 pairs), OpenRCA (3), PetShop (10).
Data: `data/rq1/pair_subsystem_deltas.csv` (layer=L1, variant=mean), `data/rq1/pairs.csv` (layer=L1, variant=mean;
pooled = `pooled_se` = equal-subsystem-weight pooled Δ; `sd_sub_delta`), `data/rq1/margin_dispersion.csv`
(`margin_lt_sd`). One row per pair labelled "A vs B" (Δ = acc_a − acc_b). On each row: one point per
subsystem at `delta` with horizontal CI bar [`lo`,`hi`]; marker shape distinguishes subsystems within the
panel (legend per panel); a short vertical tick/line at the pooled Δ. Points whose sign is opposite to the
pooled Δ drawn in a highlight colour (and filled if their CI excludes 0, hollow otherwise). Grey background
band on rows with `margin_lt_sd == True`. Text on the right of each row: "|m|=0.227 SD=0.080".
Vertical dashed line at 0. Sort rows within panel by |pooled Δ| descending.

## Figure 2 — `figures/fig2_margin_spread.pdf` (full width, two panels side by side, ~2.6 in tall)
Left panel "Our audit": `data/rq1/margin_dispersion.csv`; x = `sd_sub_delta`, y = `abs_margin`;
marker shape = family (RE1 / OpenRCA / PetShop); filled = L1, hollow = L2; colour = `reverses` (True = highlight
colour, False = neutral). Right panel "Published tables": `data/rq1/published_tables_margin_dispersion.csv`;
x = `sd_units`, y = `abs_margin`; shape = table (RCAEval Table 6 / PetShop Table 8); colour = `n_rev > 0`.
Both panels: diagonal y = x line, equal axis limits shared across panels, axis labels
"Between-subsystem SD of Δ" (right panel: "Between-stratum SD of Δ") and "|pooled margin|".
Add a small text in each panel: counts "below line: X/Y reverse; above: Z/W" computed from the data.

## Figure 3 — `figures/fig3_input_dumbbell.pdf` (full width, two panels RE1-SS | RE1-TT, ~2.2 in tall)
Data `data/rq2/input_accuracy.csv`. One row per method (display names), two markers joined by a line:
`acc_correct_input` (label "simple_data.csv (reduced table)") and `acc_submitted_input` (label "data.csv (raw export)").
x axis Acc@1 in [0,1]. Same row order in both panels (sort by SS reduced-table accuracy).

## Table 4 — `tables/tab4_main.tex` (full width, `\scriptsize`, `tabular*` with `\extracolsep{\fill}`)
Data `data/rq1/accuracy_by_subsystem.csv` (wide: method × subsystem), coverage from `data/rq2/coverage.csv`
(for seeded methods average `coverage` over seeds). Rows grouped under three sub-headings
(Published methods / Probes / Shipped baselines; `\midrule` + italic group label). Merge e_diagnosis and
epsilon_diagnosis into a single row (they never overlap). Columns grouped with `\cmidrule` headers:
RCAEval RE1 (OB SS TT) | RCAEval RE2 (OB SS TT) | OpenRCA (Bank Mkt-1 Mkt-2 Tel) | PetShop (High Low Tmp-1 Tmp-2).
Cells: 3 decimals; NaN → "--"; if coverage < 1 append `$^\dagger$`. Bold the best value per column.
Add a header row with case counts per subsystem taken from `coverage.csv` column `n`.

## Table 5 — `tables/tab5_cross_release.tex`
Data `data/rq1/cross_release.csv`, layer=L1 (18 rows = 6 pairs × 3 apps). Columns:
Pair | App | Δ RE1 [CI] | Δ RE2 [CI] | Same sign | Opposite CIs. Booleans as \checkmark or blank.
Group rows by pair (pair name only on first row of group).

## Table 6 — `tables/tab6_regret.tex`
Data `data/rq1/regret.csv`, weighting = system_equal only. One row per family × layer (8 rows):
Family | Layer ("Published" / "All") | Pooled winner | Winner pooled acc. | #subsystems with regret > 0 (out of k) |
Max regret (subsystem). Regret > 1e-9 counts as > 0. Max regret "0" if none.

## Table 7 — `tables/tab7_factors.tex`
One row per factor × family × layer. Columns: Factor | Family | Layer | Comparisons | Sign changes | Reversal-status changes.
A "comparison" = one (pair, subsystem) row. "Reversal-status change" = `rev_ref != rev_alt` (or `rev_ref != rev_rw`).
- Input representation: `data/rq2/condition_comparisons.csv`, factor=input_representation;
  EXCLUDE rows with `sub == "RE1::OB"` (RE1-OB ships only one file, so both conditions are identical).
- Release defect: same file, factor=release_defect (RE1-OB duplicate "time" header; reference = fixed, alternative = as shipped).
- Scoring (strict =1) and Scoring (≥0.5): same file, factor=scoring_strict / scoring_ge05.
- Denominator (valid outputs only): `data/rq2/denominator_valid_only.csv`.
- Fault re-weighting: `data/rq2/fault_reweighted.csv` (sign_change, rev_ref vs rev_rw).
Also print the full summary to stdout as a plain table so it can be checked.

## Acceptance
`python3 paper/scripts/make_figures.py && python3 paper/scripts/make_tables.py` runs from repo root without error,
writes all files above, and prints for each output the key counts it computed.

## Table 1 — `tab1_survey.tex` (surveyed papers, main text)

Source: `data/survey/papers.csv` (id, first author, year, venue), `per_system.csv` (per_system ∈ {yes, single_system}),
`release_check.csv` (L1 ∈ {yes, no, undeterminable}; kind ∈ {demo_case, own_method, all_methods_one_table}; L3 ∈ {no, partial, undeterminable}),
`excluded.csv` (ids without full text; dropped). One row per full-text paper (23), in `papers.csv` order.
Columns: Paper (first author et al.) | Year | Venue (ASCII; `VENUE_FIX` overrides the annotated strings) | Per-system (yes / single system) |
Per-case outputs released (-- / one demonstration case / own method, all test cases / all methods, simulation study only / release unreachable) |
All methods (no / partial / --). Plain `tabular`, `@{}lllllc@{}`.
