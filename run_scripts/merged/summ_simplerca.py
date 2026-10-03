import json, glob, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f)); s = d["summary"]
    print("%-50s n=%3d T1=%.3f T3=%.3f T5=%.3f A3=%.3f A5=%.3f MRRall=%.3f MRRhit=%.3f exc=%d empty=%d" % (
        f.split("/")[-1], s["n"], s["AC@1"], s["AC@3"], s["AC@5"], s["Avg@3"], s["Avg@5"],
        s["MRR_all_cases"], s["MRR_platform_hit_only"], s["n_exceptions"], s["n_empty"]))
