import numpy as np
import vm_placement as vp
from shift_types import make_trace

def best_fit_limit(d, limit=0.8):
    free = np.full(vp.N_HOST, limit * vp.CAP)
    a = np.zeros(vp.N_VM, int)
    for i, x in enumerate(d):
        c = [h for h in range(vp.N_HOST) if free[h] >= x]
        h = min(c, key=lambda h: free[h] - x) if c else int(free.argmax())
        a[i] = h; free[h] -= x
    return a

POLICIES = ["best_fit", "best_fit_80", "predictive"]

print(f"{'shift':<8}{'policy':<13}{'ovl after':>16}{'bal_std after':>18}{'migrations':>14}")
for mode in ["ramp", "jump", "burst"]:
    res = {p: [] for p in POLICIES}
    for s in range(15):
        vp.rng = np.random.default_rng(s)
        L = make_trace(mode)
        runs = {"best_fit": vp.run_static(L, vp.best_fit),
                "best_fit_80": vp.run_static(L, best_fit_limit),
                "predictive": vp.run_predictive(L)}
        for name, (hist, mig) in runs.items():
            m, _ = vp.metrics(L, hist, mig)
            res[name].append([m["after"]["overloads"], m["after"]["balance_std"], mig])
    for name, r in res.items():
        a = np.array(r); mu, sd = a.mean(0), a.std(0)
        print(f"{mode:<8}{name:<13}{mu[0]:>9.1f} ± {sd[0]:<5.1f}"
              f"{mu[1]:>11.3f} ± {sd[1]:<5.3f}{mu[2]:>8.1f} ± {sd[2]:<4.1f}")
    print()