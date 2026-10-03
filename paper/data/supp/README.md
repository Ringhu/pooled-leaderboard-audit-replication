# Diagnostic outputs used only by the supplement

Each file is the verbatim stdout of a script run on the audit server in `$AUDIT_ROOT/` (env `auditstack`),
fetched on 2026-09-30. None of these numbers is used in the main text; the supplement tables built from them are
written by `paper/scripts/make_supp_extra.py`.

| file | command (in `$AUDIT_ROOT/`) | inputs |
|---|---|---|
| `simplerca_summary.txt` | `python scripts/summ_simplerca.py` | `runs/simplerca_author/*.json` |
| `re1_window_compare.txt` | `python scripts/re1_window_compare.py` | `runs/baro/`, `runs/probes/` (10- and 20-minute runs) |
| `re2_table6_compare.txt` | `python scripts/re2_table6_compare.py` (full output) | `runs/re2/*.json`; Table 6 values typed from the RCAEval paper |
| `openrca_default_answers.txt` | `python scripts/openrca_default_answer_check.py` | OpenRCA prediction files of the earlier wrapper runs |

Verification notes with context: `.research/verifications/2026-09-27-{simplerca-author-g1,re1-window,re2-rcaeval-table6,openrca-wrapper-audit}.md`.
