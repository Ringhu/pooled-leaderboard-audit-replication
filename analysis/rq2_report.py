"""Render RQ2 outputs (analysis/main/rq2/) as markdown; numbers copied from the output files."""
import json

import pandas as pd

import os
D = os.environ.get("AUDIT_ROOT", "work") + "/analysis/main/rq2"
s = json.load(open(f"{D}/summary.json"))
f = json.load(open(f"{D}/fault_summary.json"))
C = pd.read_csv(f"{D}/condition_comparisons.csv")
r3 = lambda x: x.round(3).to_markdown(index=False)  # noqa: E731


def block(title, keys, src):
    print(f"\n### {title}\n")
    print(pd.DataFrame([dict(key=k, **{a: (";".join(v) if isinstance(v, list) else v) for a, v in src[k].items()})
                        for k in keys if k in src]).round(3).to_markdown(index=False))


print("## (a) 输入表示\n")
print("参照 = 主表（RE1-SS/TT simple_data.csv；RE2 simple_metrics.csv）；替代 = 两稿提交版的输入（RE1-SS/TT data.csv，"
      "RE1-OB 不变；RE2 metrics.csv）。方法集 = 两种输入都有逐案例输出的方法（RE1 无 CausalRCA）。反转按实例（方法对 × 子系统）计。")
block("汇总", ["input_RE1_L1", "input_RE1_L2", "input_RE2_L2"], s["summary"])
print("\n逐方法准确率：\n")
print(r3(pd.read_csv(f"{D}/input_accuracy.csv")))
print("\nRE1 在任一输入下反转的实例：\n")
q = C[(C.factor == "input_representation") & (C.family == "RE1") & (C.rev_alt | C.rev_ref)]
print(r3(q[["layer", "a", "b", "sub", "delta_ref", "lo_ref", "hi_ref", "pooled_ref", "rev_ref", "rev_ci_ref", "delta_alt",
            "lo_alt", "hi_alt", "pooled_alt", "rev_alt", "rev_ci_alt"]]))

print("\n## (b) 输出完整性\n")
cov = pd.read_csv(f"{D}/coverage.csv")
print("覆盖率 < 1 的 方法 × 子系统（× 种子）：\n")
print(r3(cov[cov.coverage < 1]))
block("分母口径：全部案例（参照） vs 只算有效输出（替代），点估计",
      [k for k in s["summary"] if k.startswith("denominator_")], s["summary"])
vo = pd.read_csv(f"{D}/denominator_valid_only.csv")
print("\n只因分母口径而改变反转状态的实例：\n")
print(r3(vo[vo.rev_ref != vo.rev_alt]))

print("\n## (c) 故障组成\n")
print("故障类别构成（案例数）：\n")
print(pd.read_csv(f"{D}/fault_composition.csv").to_markdown(index=False))
block("按共同故障类别等权重加权后（参照 = 全部案例）", [k for k in f["summary"] if k.startswith("fault_")], f["summary"])
block("跨版本（RE2 只保留 RE1 的故障类型）", [k for k in f["summary"] if k.startswith("cross_release")], f["summary"])
W = pd.read_csv(f"{D}/fault_reweighted.csv")
print("\n加权后反转状态改变的实例：\n")
print(r3(W[W.rev_ref != W.rev_rw][["family", "layer", "a", "b", "sub", "delta_ref", "pooled_ref", "delta_rw", "lo_rw",
                                    "hi_rw", "pooled_rw", "rev_ref", "rev_rw", "rev_ci_rw"]]))
print("\nOpenRCA 故障原因到类别的映射（Claude 编码）：\n")
print("```\n" + json.dumps(f["config"]["reason_class"], indent=1, ensure_ascii=False) + "\n```")

print("\n## (d) 评分规则（OpenRCA）\n")
block("官方部分得分（参照） vs 严格 =1 / ≥0.5 二值", [k for k in s["summary"] if k.startswith("scoring_")], s["summary"])
print("\n各规则下准确率：\n")
print(r3(pd.read_csv(f"{D}/openrca_scoring_accuracy.csv")))
q = C[C.factor.str.startswith("scoring") & (C.sign_change | (C.rev_ref != C.rev_alt))]
print("\n符号或反转状态改变的实例：\n")
print(r3(q[["factor", "layer", "a", "b", "sub", "delta_ref", "pooled_ref", "delta_alt", "pooled_alt", "rev_ref", "rev_alt",
            "sign_change"]]))

print("\n## (e) 各因素并列\n")
rows = []
for fac, g in C.groupby(["factor", "family", "layer"]):
    rows.append(dict(factor=fac[0], family=fac[1], layer=fac[2], comparisons=len(g), sign_changes=int(g.sign_change.sum()),
                     reversal_status_changes=int((g.rev_ref != g.rev_alt).sum())))
for (fam, layer), g in vo.groupby(["family", "layer"]):
    rows.append(dict(factor="denominator_valid_only", family=fam, layer=layer, comparisons=len(g),
                     sign_changes=int(g.sign_change.sum()), reversal_status_changes=int((g.rev_ref != g.rev_alt).sum())))
for (fam, layer), g in W.groupby(["family", "layer"]):
    rows.append(dict(factor="fault_reweighting", family=fam, layer=layer, comparisons=len(g),
                     sign_changes=int(g.sign_change.sum()), reversal_status_changes=int((g.rev_ref != g.rev_rw).sum())))
E = pd.DataFrame(rows)
E["sign_change_share"] = E.sign_changes / E.comparisons
print(r3(E))
T = pd.read_csv(f"{D}/kendall_tau.csv")
print("\n子系统内方法排序的 Kendall τ（参照 vs 替代）与第一名是否改变：\n")
T["top_changed"] = T.top_ref != T.top_alt
print(r3(T))
print("\n## 运行配置\n```\n" + json.dumps(s["config"], indent=1) + "\n" + json.dumps(
    {k: v for k, v in f["config"].items() if k != "reason_class"}, indent=1) + "\n```")
