import pandas as pd

print("=== Fixed results ===")
df = pd.read_csv("results/lambda_sensitivity_fixed.csv")
df = df[df["fold_id"] == "leave_condition_00"]
for _, r in df.iterrows():
    print(f"{r['model']:55s} seed={r['seed']} acc={r['accuracy']:.4f} lam={r['lambda_domain']} grl={r.get('grl_schedule','N/A')} raw={r.get('coral_no_normalize','N/A')}")

print("\n=== All probes ===")
pf = pd.read_csv("results/lambda_sensitivity_probes.csv")
pf = pf[pf["fold_id"] == "leave_condition_00"]
for _, r in pf.iterrows():
    ckpt_short = r["backbone_model"]
    print(f"{ckpt_short:55s} target={r['target']:10s} BA={r['balanced_accuracy']:.4f} norm={r['normalized_predictability']:.4f}")
