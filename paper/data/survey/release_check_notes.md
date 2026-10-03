# Release check: per-case outputs in the public release artifacts of the 26 listed RCA papers (the three papers in excluded.csv are not counted in the paper)

Check time: 2026-10-02T13:45:20Z (all checked_utc timestamps uniformly use this moment, which is the snapshot time at the end of a continuous checking session; the actual access to some repositories happened a few hours earlier, but all checks were completed on the same day, and no inconsistency arose from repository content changes in between).

## Method

Each of the 26 papers' public release artifacts (GitHub repository / Zenodo record / project homepage / anonymous-review mirror) was checked one by one, to answer three layered questions:

- **L1**: Does the release artifact contain *any* per-case prediction or score (for any method, any dataset)? yes / no / undeterminable.
- **L2** (added when L1 = yes): which methods are involved (the paper's own method vs. baselines), which datasets/systems, how many cases, and whether this matches the case count claimed in the paper.
- **L3**: Is the released per-case output sufficient to reconstruct *all* pairwise method comparisons in the paper's main table? yes / partial / no.

Classification criterion: releasing only code, only raw telemetry/fault-injection data, only summary tables, or only ground-truth labels does not count as per-case output.

**Tools and constraints**: `gh api repos/<owner>/<repo>/git/trees/<branch>?recursive=1` was used preferentially to get a recursive file-name listing; once a keyword hit was found, `gh api repos/<owner>/<repo>/contents/<path> --jq '.content' | base64 -d` was used to retrieve file content. For Zenodo, only `https://zenodo.org/api/records/<id>` was used to look at the file-name/size metadata in the `files` field; for zips too large to download, a small script was additionally written based on HTTP Range requests plus manual parsing of the ZIP central directory (EOCD), which fetches only the central-directory portion at the tail of the zip file (typically tens of KB to a few MB) and uses Python's `zipfile` to resolve the full file-name list, without downloading the zip itself. No repository was cloned at any point, and no dataset or large file was downloaded to the local machine.

**Two methodological notes, for reference during re-checking**:

1. **`raw.githubusercontent.com` became unreachable partway through this session** (IPv6 reported Network unreachable, IPv4 connections timed out), even though requests to the same host earlier in the session had succeeded (e.g., baro2024's README.md/main.py). Partway through, the check switched to `gh api .../contents/<path>` (via api.github.com) to retrieve file content, which remained stable thereafter. A few checks completed earlier (e.g., one file for causalrca2023) were obtained via the raw method while that host was still reachable and were not re-verified, but the content itself is unaffected.
2. **Some papers' full-text `.txt` files were classified as binary ("data") by the Unix `file` command** (due to control characters such as `\x0c`, presumably from the PDF-to-text conversion pipeline); for such files, running plain `grep` without `-a` **silently returns an empty result** even when there is a match. Affected files: baro2024, causalrca2023, circa2022, dynacausal2025, eadro2023, ocean2024, openrca2025, run2024, toomanycooks2025, tracediag2023, tracerca2021, tvdiag2024 — 12 `.txt` files in total. All keyword searches (github/zenodo/doi/drive.google/huggingface/anonymous.4open.science link patterns, plus keywords such as "available/release/open-source") were re-run on these 12 files using `grep -a`. The re-run results were largely consistent with before, with only two additions: openrca2025.txt confirmed the sentence "The OpenRCA code and data are available at GitHub" (consistent with the already-known conclusion about the official repository, which did not change the judgment); toomanycooks2025.txt revealed a second anonymous link that had previously been missed (`anonymous.4open.science/r/microservo`), which has been incorporated into that paper's check, with the conclusion given below.

For the 3 papers whose full text was unavailable (microrank2021, nezha2023, mrca2024), only their official/presumed-official GitHub repository itself could be checked, with no way to cross-validate against statements in the paper's original text (e.g., total case count, whether there are additional datasets); the L2/L3 judgments for these three papers are correspondingly discounted, as detailed in each entry's note.

## Summary Table (one row per paper for L1/L3, in the order they appear in the CSV)

| id | L1 | L3 | One-sentence conclusion |
|---|---|---|---|
| baro2024 | no | no | Repository is pure code; Zenodo datasets are raw telemetry + anomaly-detection intermediate caches, no prediction output |
| circa2022 | no | no | Pure code repository, no data/output |
| rcd2022 | no | no | Repository has only raw input telemetry, no prediction output |
| tracerca2021 | no | no | No full text; repository is pure code, no data/output (shallow check) |
| microrank2021 | no | no | No full text; official repository is code + paper PDF, no data/output |
| rcaeval2025 | no | no | Repository is pure code; dataset (via Zenodo) is raw data + ground truth, no predictions |
| fang2026 | no | no | Statement says "will release in the future"; the actual repository already exists but is still pure code |
| **openrca2025** | **yes** | no | Official repository contains 335 per-query predictions (exactly matching the claimed 335 faults), but covers only one model in one setting, so the full set of main-table comparisons cannot be reconstructed |
| petshop2024 | no | no | Dataset has only raw metrics + ground truth; notebook output is only aggregate accuracy |
| eadro2023 | no | no | Pure code repository, no data/output |
| nezha2023 | yes | no | No full text; official repository's log/ contains per-case predictions + comparisons in unstructured text form (own method only, no baselines) |
| mabc2024 | no | no | Repository has only statistical-chart images, no structured per-case files |
| mulan2024 | no | no | No own release link found anywhere in the full text |
| ocean2024 | no | no | No own release link found in the full text (an unrelated third-party reproduction repository exists) |
| tvdiag2024 | no | no | Logs are aggregate training metrics; data/ has only raw data + ground-truth labels |
| mrca2024 | no | undeterminable | No full text; GitHub repository is pure code; external SharePoint "AD results" link could not be checked (judged undeterminable) |
| chase2024 | undeterminable | undeterminable | Code released via a Google Drive link, which could not be checked without downloading |
| howfar2024 | no | no | Zenodo zip (checked via remote listing) contains synthetic-data-generator output + ground truth, no predictions |
| lemmarca2024 | no | no | GitHub is pure code; project homepage/Zenodo both have only data/metadata, no results |
| sparserca2024 | no | no | No own release link found anywhere in the full text |
| tracecontrast2024 | no | no | No own release link found anywhere in the full text |
| tracediag2023 | no | no | No own release link found anywhere in the full text (Microsoft industry paper) |
| run2024 | no | no | Pure code repository, no data/output |
| dynacausal2025 | no | no | No own release link found anywhere in the full text |
| causalrca2023 | yes | no | Official repository (renamed to CausalRCA_code) has per-case ranked predictions + scores under pa_result/ (own method only, no baselines) |
| toomanycooks2025 | undeterminable | undeterminable | Both anonymous-review links are now invalid (one repo_not_found, one repository_expired), content could not be checked |

**Summary counts**:
- L1: yes = 3 (openrca2025, nezha2023, causalrca2023); no = 21; undeterminable = 2 (chase2024, toomanycooks2025).
- L3: yes = 0; partial = 0; no = 23; undeterminable = 3 (mrca2024, chase2024, toomanycooks2025, all because an external link could not be checked, not because checking revealed a partial case).

## Per-Paper Evidence

**baro2024** (GitHub `phamquiluan/baro`@`e35f4ec`; Zenodo 11094092, 11046533): the repository's tree listing confirms it contains only code files (README/main.py/reproducibility.py, etc.); the two Zenodo records are a software-archive zip (code) and three dataset zips (fse-ob/fse-ss/fse-tt) respectively — checking the file-name listings confirms the content is raw telemetry data plus an anomaly-detection-stage cache file (of the `naive_bocpd.json` kind), not root-cause ranking/prediction results. L1 = no.

**circa2022** (`NetManAIOps/CIRCA`@`0215e18`): pure code repository; the `img/output` directory is empty or gitignored; no data or output files of any kind. L1 = no.

**rcd2022** (`azamikram/rcd`@`373882c`): the repository contains raw input telemetry such as `sock-shop-data/*/{anomalous.csv,normal.csv}`; no prediction/ranking output files. L1 = no.

**tracerca2021** (no full text; `NetManAIOps/TraceRCA`@`b9087c7`): the repository contains only scripts such as `run_*.py`, no data or output files. Only a repository file-name listing and a README-level check were performed; it was not verified script-by-script whether result files might be generated at runtime and separately uploaded by a user afterward — this is the shallowest check performed in this round. L1 = no (shallow check).

**microrank2021** (no full text; `IntelligentDDS/MicroRank`@`efd85e8`): the official repository is code plus the paper PDF, no data or output files. L1 = no.

**rcaeval2025** (`phamquiluan/RCAEval`@`259ea41`; Zenodo 13305663, shared with howfar2024): all 149 files in the repository are code, no data/output directory; the dataset is distributed via Zenodo, with content described in the howfar2024 entry (raw data + ground truth, no predictions). L1 = no.

**fang2026** (`LGU-SE-Internal/Simple-RCA`@`1be3911`, orig branch): the paper's data-availability statement is in future tense ("we promise to release"); at the time of checking, the repository already exists, but the orig branch is still pure code, with no result files. Note: it was found that RCAEval actually uses the `rcaeval`/`rcaeval_re3_ss` branches of the same repository rather than the default orig branch, which is inconsistent with both the wording of this paper's data-availability statement and the actual code. L1 = no.

**openrca2025** (`microsoft/OpenRCA`@`c1bd4af`): the four files `rca/archive/agent-{Bank,Market-cloudbed-1,Market-cloudbed-2,Telecom}.csv` each contain, row by row, prediction, groundtruth, pass/fail, and score fields. Counted precisely using Python's `csv.DictReader` (rather than the unreliable `wc -l`, since the prediction column embeds multi-line JSON strings): 136+70+78+51 = 335, exactly matching the 335 faults claimed in the paper's abstract. L1 = yes. L2: method is RCA-agent only (Claude 3.5 Sonnet implementation), datasets Bank/Market-cloudbed-1/Market-cloudbed-2/Telecom, 335 cases in total. L3 = no — the paper's main table compares multiple models (Claude 3.5, GPT-4o, etc.) across multiple settings (Base vs. Agent, oracle vs. balanced sampling); the released artifact covers only one model in one setting, so the full set of pairwise comparisons in the main table cannot be reconstructed from it.

**petshop2024** (`amazon-science/petshop-root-cause-analysis`@`2e96f93`): under `dataset/`, each issue has only `metrics.csv` (raw metrics) and `target.json` (ground truth); `code/RCA-Dataset-Eval.ipynb` (downloaded via `gh api contents` + base64 and inspected for cell outputs using Python's `json` module) shows that the notebook computes accuracy on the fly via `rca_task.evaluate(...)`, and its saved cell outputs are only aggregate print statements (e.g., "top1 got 0.571") — no per-case prediction table is persisted. L1 = no.

**eadro2023** (`BEbillionaireUSD/Eadro`@`82ff9c9`): pure code repository (`codes/*.py`), no data or output files. L1 = no.

**nezha2023** (no full text; `IntelligentDDS/Nezha`@`d814010`): the 4 pre-generated run logs under `log/` (`OnlineBoutique_service_result.log`, `OnlineBoutique_innerservice_result.log`, `Trainticket_service_result.log`, `Trainticket_innerservice_Result.log`) record, in unstructured text form, each injected fault case's comparison between Nezha's own ranked predictions and the ground truth, along with aggregate metrics such as AC@1/3/5. Because the full text is unavailable, it could not be verified whether the paper involves a third system; the "Fault number" counts in the logs (e.g., 56 cases under one configuration) and the format itself are also not a clean structured file. L1 = yes. L2: method is Nezha's own only (no per-case files found for baseline methods), datasets OnlineBoutique and TrainTicket, case count could not be given as a unified total across configurations. L3 = no — only the method's own output is present, with no baseline method's per-case data for comparison.

**mabc2024** (`zwpride/mABC`@`79272ce`): the repository contains `result.png` and several statistical-chart images (image format) under `handle/info/figs`, but no structured per-case prediction files (csv/json, etc.). L1 = no.

**mulan2024**: a full-text keyword search (github/zenodo/doi/drive.google/anonymous.4open.science, etc.) found no own release link at all; no external verification was undertaken. L1 = no.

**ocean2024**: a full-text search (including an `-a` re-search for the file flagged as binary) found no link to the authors' own repository/dataset; there is a third-party unofficial reproduction repository on GitHub unrelated to this paper's authors, which is not counted within the scope of checking official release artifacts. L1 = no.

**tvdiag2024** (`WHU-AISE/TVDiag`@`cf6f334`): `logs/{gaia,hotel,sockshop}/*.log` (downloaded via `gh api contents` and inspected) contain only aggregate training metrics such as HR@1-5/MRR/precision/recall/F1; under `data/` there are only raw pkl/json files and `label.csv` (ground-truth labels), no per-case prediction files. L1 = no.

**mrca2024** (no full text, ACM paywall; `ddd87613/MRCA`@`c507790`, confirmed as the corresponding repository only by matching the repository README against the paper title, without full-text cross-validation): the repository itself is pure code, no data/output files. The "AD results" (anomaly-detection-stage results) mentioned in the README link out to a Microsoft SharePoint shared folder, and no available API could check its file list or content; judging from the naming and the paper's pipeline structure, this should be the output of the intermediate anomaly-detection stage rather than the final root-cause localization prediction, but this could not be independently confirmed. L1 = no (for the repository itself), L3 = undeterminable (because the external link could not be checked, it is uncertain whether broader-coverage per-case output exists).

**chase2024**: the paper states "we release the source code" and gives a Google Drive file link (not a repository directory structure, so its content cannot be listed with any API). A `curl -sIL` HEAD/redirect request against this link hung indefinitely; the background process was terminated. Under the constraint of not downloading data, its content could not be checked. L1 = undeterminable, L3 = undeterminable.

**howfar2024** (`phamquiluan/RCAEval`@`259ea41`; Zenodo 13305663): this Zenodo record contains several large zips (`rca_circa.zip`, `sock-shop-1.zip`, etc.); a custom HTTP Range request + ZIP central-directory parsing script (see Method section) was used to list the file names (without downloading the zip itself); typical paths include `rca_circa/*/data.csv`, `rca_circa/*/root_cause.txt`, `rca_circa/*/inject_time.txt`, `online-boutique/*/data.csv`, `sock-shop-1/*/data.csv`, `sock-shop-2/*/data.csv`, `train-ticket/*/data.csv`. The content is the raw output of a synthetic data generator (`data.csv`) and fault-injection ground-truth labels (`root_cause.txt`/`fe_service.txt`/`inject_time.txt`/`details.pk`), consistent with the paper's statement that "`rca_circa`/`rca_rcd` are datasets synthesized using CIRCA/RCD's data-generation method" — containing no method's prediction results. L1 = no.

**lemmarca2024** (`zhengzhangchen/LEMMA_rca_baselines`@`9b56afb`; project homepage `lemma-rca.github.io`; Zenodo 20516735): all 129 files in the GitHub repository are baseline-method code, no result files; the project homepage only offers raw/preprocessed data downloads, no results; the `files` field of the Zenodo record contains only documentation-type files such as README/dataset_card/access_instructions/metadata, no actual data or result files. L1 = no.

**sparserca2024**: a full-text search found no own release link at all. L1 = no.

**tracecontrast2024**: a full-text search found no own release link at all. L1 = no.

**tracediag2023**: a full-text search (including an `-a` re-search) found no own release link at all; this is a Microsoft industry paper, with no evidence of publicly released code. L1 = no.

**run2024** (`zmlin1998/RUN`@`e2c97d7`): pure code repository (`RUN.py`, `model.py`, `data_provider/`, etc.), no data or output files. L1 = no.

**dynacausal2025**: a full-text search (including an `-a` re-search) found no own release link at all. L1 = no.

**causalrca2023** (the original repository name `CausalRCA` now returns 404 on GitHub; the renamed `AXinx/CausalRCA_code`@`21f4eb0` was located via `gh api search/repositories`): `pa_result/{all_service,single_service,latency}/*.txt` stores per-case root-cause ranking lists and PC/pagerank scores, stratified by service/fault type (cpu/memory/network)/hyperparameter (eta100/1000, gamma5/75); `data_collected/*.pkl` is the raw collected per-fault-type metrics (input data, not output). L1 = yes. L2: method is CausalRCA's own only (no per-case files found for baseline methods), dataset Sock Shop, cases form multiple result sets combining fault type x service x hyperparameter, not individually cross-checked against the row count of the paper's main table. L3 = no — only the method's own results are present, with no baseline method's per-case comparison.

**toomanycooks2025**: the paper's main text mentions two `anonymous.4open.science` double-blind-review links — one hosting reproduction code for four baseline methods (PDiagnose/CloudRCA/RMLAD/TrinityRCL) (`Empirical-Study-on-Multisource-Failure-Diagnosis-49F0`), and one hosting a newly created MicroServo dataset (`microservo`, a link that was only discovered when the `-a` re-search was performed on the file flagged as binary — the single new finding in this round that resulted from that methodological note). At check time, the API for both links returned errors: the former `{"error":"repo_not_found"}`, the latter `{"error":"repository_expired"}` — the double-blind review period has ended, and both anonymous links are now invalid, so their content could not be checked. A `gh api search/repositories` search on the paper's title keywords and on `org:AIOps-Lab-NKU` (which lists 30 repositories) was also attempted, with no obviously corresponding non-anonymous permanent mirror found. L1 = undeterminable, L3 = undeterminable.

## List of Incompletely Checked / Undeterminable Items

- **chase2024**: code released via a single Google Drive file link; could not be checked under the no-download constraint, recorded as undeterminable.
- **toomanycooks2025**: both anonymous-review links are now invalid (repo_not_found / repository_expired); content could not be checked at all, recorded as undeterminable.
- **mrca2024**: the GitHub repository itself was confirmed as no, but the external Microsoft SharePoint "AD results" link cited in its README could not be checked through any available API, recorded as undeterminable; also, this paper's full text is unavailable, so the correspondence was confirmed only by matching the title against the repository README, without cross-validation against the original text.
- **tracerca2021**: full text unavailable; only the official repository's file-name listing and README were checked; it was not verified script-by-script whether there are result files generated only at runtime and not included in the repository — this is the shallowest check performed in this round.
- **microrank2021, nezha2023**: full text likewise unavailable; the checking scope was limited to the presumed-official repository itself, with no way to cross-validate against the case count/dataset count in the paper's original text; nezha2023's log format is unstructured text, so a unified total case count across configurations could not be given.
- **Methodological note**: the 12 paper full-text `.txt` files misclassified as binary by the `file` command were re-searched with `grep -a` (see the Method section), confirming that no other key information was missed apart from toomanycooks2025's second anonymous link; however, a small probability cannot be ruled out that some release-channel phrasing is still not covered by the current keyword list.
- This check does not address at all whether the numbers in the papers themselves are correct, or other known issues with benchmarks such as RCAEval/OpenRCA; it answers only the single factual question of "whether the release artifact contains per-case output."
