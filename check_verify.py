import pandas as pd
df = pd.read_csv("results/verify_srt_base.csv")
for _, r in df.iterrows():
    print(f"{r['model']:45s} seed={r['seed']} acc={r['accuracy']:.4f} lam={r['lambda_domain']} grl={r.get('grl_schedule','N/A')} raw={r.get('coral_no_normalize','N/A')}")
