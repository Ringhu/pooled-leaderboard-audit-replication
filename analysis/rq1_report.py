"""Render RQ1 outputs (analysis/main/rq1/) as markdown tables; numbers are copied from the CSVs, not retyped."""
import json

import pandas as pd

import os
D = os.environ.get("AUDIT_ROOT", "work") + "/analysis/main/rq1"
P = pd.read_csv(f"{D}/pairs.csv")
S = pd.read_csv(f"{D}/pair_subsystem_deltas.csv")
r3 = lambda x: x.round(3).to_markdown(index=False)  # noqa: E731

print("## A. 反转计数（三种子平均；系统等权 pooled；括号外=点估计反转对数，CI=子系统 CI 支持的反转对数）\n")
g = P[P.variant == "mean"].groupby(["family", "layer"]).agg(
    pairs=("a", "size"), rev_pairs=("n_rev_se", lambda x: int((x > 0).sum())),
    rev_ci_pairs=("n_rev_ci_se", lambda x: int((x > 0).sum())),
    rev_pairs_cw=("n_rev_cw", lambda x: int((x > 0).sum())),
    rev_ci_pairs_cw=("n_rev_ci_cw", lambda x: int((x > 0).sum())),
    pi_computed=("re_pi_lo", lambda x: int(x.notna().sum())), pi_crosses_0=("pi_crosses_0", lambda x: int((x == True).sum())),  # noqa: E712
    separation=("separation", "sum"), lrt_holm_sig=("lrt_p_holm", lambda x: int((x < .05).sum())),
    wald_holm_sig=("wald_p_holm", lambda x: int((x < .05).sum()))).reset_index()
print(g.to_markdown(index=False))
print("\n（RE2 只作描述，不计入 k；separation = 某子系统中一方命中率为 0 或 1，GLM 检验不可靠）\n")

cols = ["a", "b", "k", "pooled_se", "pooled_se_lo", "pooled_se_hi", "pooled_cw", "rev_se", "rev_ci_se", "sd_sub_delta",
        "re_delta", "re_pi_lo", "re_pi_hi", "I2", "lrt_p_holm", "wald_p_holm", "separation"]
for fam in ("RE1", "openrca", "petshop"):
    print(f"\n### {fam} 第 1 层（已发表 vs 已发表）\n")
    print(r3(P[(P.family == fam) & (P.layer == "L1") & (P.variant == "mean")][cols]))
    print(f"\n{fam} 第 1 层子系统 Δ：\n")
    print(r3(S[(S.family == fam) & (S.layer == "L1") & (S.variant == "mean")][
        ["a", "b", "sub", "n", "acc_a", "acc_b", "delta", "lo", "hi"]]))

print("\n### 第 2 层（含探针）中出现反转的方法对\n")
q = P[(P.variant == "mean") & (P.layer == "L2") & (P.n_rev_se > 0) & P.family.isin(["RE1", "openrca", "petshop"])]
print(r3(q[["family", "a", "b", "pooled_se", "pooled_se_lo", "pooled_se_hi", "rev_se", "rev_ci_se", "sd_sub_delta"]]))

print("\n## A2. 多种子方法逐种子（第 1 层，含 rcd/causalrca 的对）\n")
v = P[(P.layer == "L1") & (P.variant != "mean")]
w = v.pivot_table(index=["family", "a", "b"], columns="variant", values=["pooled_se"], aggfunc="first").round(3)
w.columns = [f"pooled_{c[1]}" for c in w.columns]
for col in ("rev_se", "rev_ci_se"):
    x = v.pivot_table(index=["family", "a", "b"], columns="variant", values=col, aggfunc="first")
    for s in x.columns:
        w[f"{col}_{s}"] = x[s]
print(w.reset_index().to_markdown(index=False))

print("\n## B. 边际 vs 离散度（三种子平均，系统等权；11 子系统家族）\n")
print(pd.read_csv(f"{D}/margin_dispersion_summary.csv").to_markdown(index=False))
md = pd.read_csv(f"{D}/margin_dispersion.csv")
print("\n反转对（按 |边际| 升序）：\n")
print(r3(md[md.reverses].sort_values("abs_margin")[["family", "layer", "a", "b", "abs_margin", "sd_sub_delta", "n_rev_se",
                                                    "n_rev_ci_se"]]))
print("\n|边际| < SD 但不反转：\n")
print(r3(md[~md.reverses & md.margin_lt_sd][["family", "layer", "a", "b", "abs_margin", "sd_sub_delta"]]))
print("\n|边际| ≥ SD 但反转：\n")
print(r3(md[md.reverses & ~md.margin_lt_sd][["family", "layer", "a", "b", "abs_margin", "sd_sub_delta"]]))
pub = pd.read_csv(f"{D}/published_tables_margin_dispersion.csv")
print("\n已发表表格（只用论文表中数字；单位 = 故障类型 / 分层）：\n")
print(pub.groupby(["table", "margin_lt_sd"]).agg(pairs=("a", "size"),
                                                reverse=("n_rev", lambda x: int((x > 0).sum()))).reset_index().to_markdown(index=False))
print("\n已发表表格中 |边际| ≥ SD 仍反转的对：\n")
print(r3(pub[(pub.n_rev > 0) & ~pub.margin_lt_sd]))
print("\n已发表表格相邻名次差（按各单位均值排序）：\n")
print(r3(pd.read_csv(f"{D}/published_tables_adjacent_gaps.csv")))

print("\n## C. 同应用跨版本（RE1 → RE2，三种子平均）\n")
cr = pd.read_csv(f"{D}/cross_release.csv")
print(cr.groupby("layer").agg(comparisons=("a", "size"), sign_agree=("sign_agree", "sum"),
                              ci_opposite=("ci_opposite", "sum")).reset_index().to_markdown(index=False))
print("\n第 1 层逐项：\n")
print(r3(cr[cr.layer == "L1"].drop(columns="layer")))

print("\n## D. pooled 第一名的 regret\n")
rg = pd.read_csv(f"{D}/regret.csv")
print(r3(rg.groupby(["family", "layer", "weighting"]).agg(
    winner=("winner", "first"), winner_pooled=("winner_pooled", "first"), runner_up=("runner_up", "first"),
    runner_up_pooled=("runner_up_pooled", "first"), k=("regret", "size"), subsystems_with_regret=("regret", lambda x: int((x > 0).sum())),
    max_regret=("regret", "max")).reset_index()))
print("\nregret > 0 的子系统：\n")
print(r3(rg[rg.regret > 0][["family", "layer", "weighting", "subsystem", "n", "winner", "winner_acc", "best_method",
                            "best_acc", "regret"]]))

print("\n## E. 小样本敏感性（只保留 n ≥ 阈值的子系统，系统等权）\n")
ss = pd.read_csv(f"{D}/small_sample.csv")
print(ss.groupby(["n_min", "family", "layer"]).agg(pairs=("a", "size"), rev=("n_rev", lambda x: int((x > 0).sum())),
                                                   rev_ci=("n_rev_ci", lambda x: int((x > 0).sum()))).reset_index().to_markdown(index=False))
print("\n## 运行配置\n")
print("```\n" + json.dumps(json.load(open(f"{D}/config.json")), indent=1) + "\n```")
print("```\n" + json.dumps(json.load(open(f"{D}/extra_summary.json"))["config"], indent=1) + "\n```")
