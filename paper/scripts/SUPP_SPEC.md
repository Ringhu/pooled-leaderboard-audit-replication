# Supplementary tables spec (paper/scripts/make_supp_tables.py → paper/supp/tables/*.tex)

Same conventions as paper/scripts/SPEC.md (display names, number formats, never hard-code results; read only paper/data/).
Every table is a `longtable` (package longtable) with booktabs rules, \footnotesize, a caption (\caption{...}\\ in the
longtable head, repeated head on continuation pages), and a \label{supp:<name>}. Output one .tex file per table.
Also write paper/supp/tables/index.tex that \input's all of them in the order below, each preceded by
\subsection*{<title>} and one sentence saying what the table shows and which CSV it comes from.

S1 `s1_pairs.tex` — all pairs, main analysis. data/rq1/pairs.csv, variant == "mean". Group by family (RE1, RE2, OpenRCA, PetShop)
   then layer (Published, All). Columns: Pair (A vs B) | k | pooled Δ (subsystems equal) [CI] | pooled Δ (case-weighted) |
   #rev | #rev CI | SD of Δ_s | below line (|pooled| < SD; mark ✓) | 95% PI (only if re_pi_lo not NaN) | LRT p (Holm) | sep. (✓ if separation).
   Use pooled_se, pooled_se_lo/hi, pooled_cw, n_rev_se, n_rev_ci_se, sd_sub_delta, re_pi_lo/hi, lrt_p_holm, separation.
S2 `s2_seeds.tex` — per-seed results: pairs.csv rows with variant in seed0/seed1/seed2. Columns: family | layer | pair | seed |
   pooled Δ [CI] | #rev | #rev CI.
S3 `s3_subsystem_deltas.tex` — data/rq1/pair_subsystem_deltas.csv, variant == "mean", layer == L1 (all four families).
   Columns: family | pair | subsystem | n | acc A | acc B | Δ [CI]. Mark rows whose Δ sign is opposite to the pair's pooled_se
   (from pairs.csv) with $^\ast$.
S4 `s4_small_sample.tex` — data/rq1/small_sample.csv: for each n_min, family, layer: pairs | pairs with ≥1 reversal | with CI-supported reversal.
S5 `s5_coverage.tex` — data/rq2/coverage.csv, all rows: subsystem | method | seed | n | valid | coverage | acc (all cases) | acc (valid only).
S6 `s6_input.tex` — data/rq2/input_accuracy.csv: subsystem | method | acc reduced/simple input | acc raw input | files.
S7 `s7_openrca_scoring.tex` — data/rq2/openrca_scoring_accuracy.csv: subsystem | method | partial credit | strict (=1) | ≥0.5.
S8 `s8_release_defect.tex` — data/rq2/release_defect_accuracy.csv (all columns) and, below it as a second longtable,
   data/rq2/condition_comparisons.csv rows with factor == "release_defect" and (sign_change or rev_ref != rev_alt):
   layer | pair | subsystem | Δ fixed [CI] | Δ as shipped [CI].
S9 `s9_published_tables.tex` — data/rq1/published_tables_margin_dispersion.csv: table | pair | k | pooled | |margin| | SD | below line | #strata reversing.
S10 `s10_fault_composition.tex` — data/rq2/fault_composition.csv (case counts per fault class, drop all-zero columns) and
   data/rq2/fault_reweighted.csv rows with sign_change == True: family | layer | pair | subsystem | Δ | Δ re-weighted [CI].
S11 `s11_survey.tex` — data/survey/papers.csv joined with data/survey/final_coding.csv (pivot item → value):
   id | year | venue (shorten to ≤ 30 chars) | Q1 | Q2 | Q3 | Q4. Title in a second line in \scriptsize italics is fine.
S12 `s12_cross_release.tex` — data/rq1/cross_release.csv, layer == L2 (all pairs): pair | app | Δ RE1 [CI] | Δ RE2 [CI] | same sign | CIs opposite.

Acceptance: `python3 paper/scripts/make_supp_tables.py` from repo root writes all files and prints row counts; a test document
/tmp/supptest/test.tex (\documentclass{article}, packages booktabs, longtable, amssymb, amsmath, geometry margin 2cm,
\newcommand{\checkmark}{$\surd$} only if not defined) that \input's paper/supp/tables/index.tex compiles with pdflatex without errors.
