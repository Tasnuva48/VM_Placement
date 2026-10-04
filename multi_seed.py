import numpy as np
import vm_placement as vp

SEEDS = range(15)
POLICIES = ["first_fit", "best_fit", "predictive"]
rows = {p: [] for p in POLICIES}

for s in SEEDS:
    vp.rng = np.random.default_rng(s)
    L = vp.gen_trace()
    runs = {"first_fit": vp.run_static(L, vp.first_fit),
            "best_fit": vp.run_static(L, vp.best_fit),
            "predictive": vp.run_predictive(L)}
    for name, (hist, mig) in runs.items():
        m, _ = vp.metrics(L, hist, mig)
        rows[name].append([m["before"]["overloads"], m["after"]["overloads"],
                           m["after"]["balance_std"], mig])

print(f"{'policy':<12}{'ovl before':>16}{'ovl after':>16}{'bal_std after':>18}{'migrations':>14}")
for p in POLICIES:
    a = np.array(rows[p])
    mu, sd = a.mean(axis=0), a.std(axis=0)
    print(f"{p:<12}{mu[0]:>9.1f} ± {sd[0]:<5.1f}{mu[1]:>9.1f} ± {sd[1]:<5.1f}"
          f"{mu[2]:>11.3f} ± {sd[2]:<5.3f}{mu[3]:>8.1f} ± {sd[3]:<4.1f}")